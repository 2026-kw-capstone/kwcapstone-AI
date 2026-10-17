"""Read previous state -> coach/verify -> atomic append/rebuild; design runs separately."""
import logging
import json
import time
import uuid
from functools import lru_cache
from services.learning.models import (LearningConfig, Observation, CoachInput, DesignInput,
                                      ANALYSIS_VERSION, POLICY_VERSION, learner_key, utc_now)
from services.learning.repository import PostgresLearningRepository
from services.learning.evidence import assess_quality, target_counts
from services.learning.agents import FeedbackCoach, PracticeDesigner
from services.learning.embeddings import OpenAIEmbedder

logger = logging.getLogger(__name__)


def log_failure(stage, exc):
    # Exception strings can contain prompts, signed S3 URLs or tokens. Never log them.
    logger.warning("learning_failure stage=%s error_type=%s", stage, type(exc).__name__)


def public_trace(trace):
    result = {k: v for k, v in trace.items() if k != "started"}
    if "verification" in result:
        result["verification"] = public_trace(result["verification"])
    return result


class LearningWorkflow:
    def __init__(self, config=None, repository=None, coach=None, designer=None, embedder=None):
        self.config = config or LearningConfig()
        self._repository = repository
        self._coach = coach
        self._designer = designer
        self._embedder = embedder

    @property
    def repository(self):
        if self._repository is None:
            self._repository = PostgresLearningRepository(config=self.config)
        return self._repository

    def process_practice(self, key, context, reference, stt, acoustic, words, insertions,
                         voice, pronunciation_score, meaning_score, meaning_feedback, pronunciation_feedback,
                         evaluation_trace=None, fallback_pronunciation=None):
        base = {"meaningDeliveryFeedback": meaning_feedback, "pronunciationFeedback": pronunciation_feedback}
        key = learner_key(key)
        if not key:
            return base
        started = time.monotonic()
        telemetry = {"mode": self.config.mode, "evaluation": evaluation_trace or {}, "coach": {}}
        quality = assess_quality(stt, acoustic, words, voice, self.config)
        method = "g2p_syllable_phoneme" if words and all(w.get("g2pAvailable") for w in words) else "grapheme_fallback"
        versions = {"analysis": ANALYSIS_VERSION, "policy": POLICY_VERSION, "agent_model": self.config.model,
                    "stt_model": "whisper-medium", "acoustic_model": "kresnik/wav2vec2-large-xlsr-korean",
                    "reference_model": "gpt-4o-mini", "feedback_model": "gpt-4o-mini",
                    "reference_source": "inferred_from_current_stt", "g2p": "g2pk",
                    "quality_config": {k: getattr(self.config, k) for k in ("silence_limit", "disagreement_limit")}}
        embedder = self._embedder or OpenAIEmbedder(self.config, telemetry)
        def search(targets):
            if not targets:
                return []
            try:
                error_types = sorted({w["errorType"] for w in words if w["errorType"] != "equal"})
                query = {"targets": targets, "error_types": error_types, "context": context}
                vector = embedder([json.dumps(query, ensure_ascii=False)])[0]
                return self.repository.search_feedback(key, vector, targets, error_types, "pronunciation")
            except Exception as exc:
                log_failure("vector_search", exc)
                telemetry["coach"]["search_failure"] = type(exc).__name__
                return []
        state = None
        try:
            # Snapshot excludes this observation, even if coach takes multiple tool turns.
            state = self.repository.get_state(key)
        except Exception as exc:
            log_failure("history_read", exc)
            telemetry["history_failure"] = type(exc).__name__
        result = base
        if state is None and self.config.mode != "baseline" and fallback_pronunciation:
            result = dict(base, pronunciationFeedback=fallback_pronunciation())
        if state is not None and self.config.mode != "baseline":
            # Meaning claims are constrained too: quote recognition evidence, not an
            # unverified assessment of intent or a personal characteristic.
            verified_base = dict(base)
            verified_base["meaningDeliveryFeedback"] = (
                f"이번에 인식된 말은 ‘{stt}’예요. 연습 상황에 필요한 핵심 내용을 담아 다시 말해 보세요."
                if stt.strip() else "인식된 말이 없어 의미 전달을 확인하기 어려워요. 다시 녹음해 주세요.")
            priority = next(iter(state.get("priorities", [])), {}).get("target")
            current = CoachInput(words=words, insertions=insertions, quality=quality, base_feedback=verified_base, priority=priority,
                                 context=context, versions=versions, reference_text=reference, method=method)
            try:
                coach = self._coach or FeedbackCoach(self.config)
                result = coach.run(current, state, telemetry["coach"], search=search)
            except Exception as exc:
                log_failure("coach", exc)
                telemetry["coach"]["failure"] = type(exc).__name__
                # Existing nonpersonal response remains usable on infrastructure failure.
                result = dict(base, pronunciationFeedback=fallback_pronunciation()) if fallback_pronunciation else base
        if self.config.mode == "baseline" and fallback_pronunciation:
            result = dict(base, pronunciationFeedback=fallback_pronunciation())
        telemetry["learning_latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        observation = Observation(
            record_id=str(uuid.uuid4()), learner_key=key, processed_at=utc_now(), context=context,
            reference_text=reference, stt_text=stt, acoustic_text=acoustic,
            method=method,
            fallback=not bool(acoustic) or any(not w.get("g2pAvailable") for w in words),
            words=words, insertions=insertions, targets=target_counts(words),
            pronunciation_score=pronunciation_score, meaning_score=meaning_score, voice=voice,
            feedback=result, quality=quality,
            versions=versions,
            telemetry=public_trace(telemetry))
        try:
            self.repository.append(observation)
            try:
                self.repository.attach_embeddings(key, observation.record_id, embedder)
            except Exception as exc:
                log_failure("embedding_enrichment", exc)
                telemetry["embedding_failure"] = type(exc).__name__
            telemetry["total_learning_latency_ms"] = round((time.monotonic() - started) * 1000, 2)
            self.repository.update_telemetry(key, observation.record_id, public_trace(telemetry))
        except Exception as exc:
            log_failure("observation_atomic_save", exc)
        logger.info("learning_practice mode=%s quality=%s elapsed_ms=%s model_calls=%s tool_calls=%s",
                    self.config.mode, quality.status, telemetry["learning_latency_ms"],
                    telemetry["coach"].get("model_calls", 0), telemetry["coach"].get("tool_calls", 0))
        return result

    def design(self, key, context, goal, baseline, levels=None, level_index=0, step_index=0):
        key = learner_key(key)
        if not key or self.config.mode != "full":
            return baseline()
        trace = {}
        started = time.monotonic()
        plan = {"target": None, "difficulty": "default", "reason": "insufficient_data"}
        result = None
        try:
            state = self.repository.get_state(key)
            if state.get("priorities"):
                plan = dict(state["priorities"][0])
                plan["recent_examples"] = state["targets"][plan["target"]]["recent"][:3]
                plan["candidates"] = [dict(p, recent_examples=state["targets"][p["target"]]["recent"][:3])
                                      for p in state["priorities"][:3]]
                request = DesignInput(context=context, goal=goal, plan=plan, levels=levels,
                                      level_index=level_index, step_index=step_index)
                designer = self._designer or PracticeDesigner(self.config)
                result = designer.run(request, state, trace)
        except Exception as exc:
            log_failure("design", exc)
            trace["failure"] = type(exc).__name__
        if result is None:
            result = baseline()
        trace["baseline"] = getattr(baseline, "metrics", {})
        trace["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        logger.info("learning_design mode=%s model_calls=%s tool_calls=%s elapsed_ms=%s fallback=%s",
                    self.config.mode, trace.get("model_calls", 0), trace.get("tool_calls", 0),
                    trace["latency_ms"], "failure" in trace)
        try:
            self.repository.record_design(key, plan, public_trace(trace))
        except Exception as exc:
            log_failure("design_save", exc)
        return result


@lru_cache(maxsize=1)
def get_learning_workflow():
    return LearningWorkflow()
