from difflib import SequenceMatcher
from .models import Quality
from services.score_service import g2p_evidence, split_korean_chars


def assess_quality(stt, acoustic, words, voice, config):
    reasons = []
    pause = voice.get("silenceRatio", {}).get("pausePercent")
    speed = voice.get("speechRate", {})
    if not split_korean_chars(stt):
        return Quality(status="unmeasurable", reasons=["empty_stt"])
    if pause is not None and pause >= config.silence_limit:
        return Quality(status="unmeasurable", reasons=["silence"])
    if speed.get("label") == "측정 불가":
        return Quality(status="unmeasurable", reasons=["no_voiced_duration"])
    if pause is None:
        reasons.append("missing_voice_evidence")
    if not acoustic:
        reasons.append("acoustic_fallback")
    else:
        stt_g, ac_g = g2p_evidence(stt), g2p_evidence(acoustic)
        ratio = SequenceMatcher(None, stt_g["spoken"], ac_g["spoken"], autojunk=False).ratio()
        if 1 - ratio > config.disagreement_limit:
            reasons.append("recognizer_disagreement")
    if not words:
        return Quality(status="unmeasurable", reasons=["empty_reference"])
    if any(not x.get("mappingReliable") for x in words):
        reasons.append("ambiguous_g2p_mapping")
    if any(not x.get("g2pAvailable") for x in words):
        reasons.append("g2p_fallback")
    return Quality(status="needs_review" if reasons else "sufficient", reasons=reasons or ["aligned_dual_recognition"])


def target_counts(words):
    """Only aligned current utterance evidence. Never read feedback or scenario plans.

    A completely absent syllable is an omission event, not an attempted phoneme.
    For substitutions count attempted source positions, including a missing coda
    within an otherwise spoken syllable. The intended reference is inferred.
    """
    counts = {}
    for word in words:
        if not word.get("mappingReliable") or word.get("hypIndex") is None:
            continue
        for position, phoneme in (word.get("spokenRefParts") or {}).items():
            if not phoneme:
                continue
            key = f"{position}:{phoneme}"
            item = counts.setdefault(key, {"attempts": 0, "errors": 0})
            item["attempts"] += 1
            item["errors"] += int(word["phonemeDiff"].get(position) == "different")
    return counts


def contains_target(text, target):
    if not target:
        return True
    evidence = g2p_evidence(text)
    if not evidence["available"] or not evidence["mapped"]:
        return False
    from services.score_service import text_to_phoneme_sequence
    return any(f'{x["position"]}:{x["phoneme"]}' == target
               for x in text_to_phoneme_sequence(evidence["text"]))
