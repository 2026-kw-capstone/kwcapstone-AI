import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

ANALYSIS_VERSION = "syllable-alignment-v2"
POLICY_VERSION = "learning-rules-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def learner_key(value):
    return value.strip() if value and value.strip() else None


@dataclass(frozen=True)
class LearningConfig:
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("LEARNING_DATABASE_URL"))
    embedding_model: str = field(default_factory=lambda: os.getenv("LEARNING_EMBEDDING_MODEL", "text-embedding-3-small"))
    similarity_min: float = field(default_factory=lambda: float(os.getenv("LEARNING_SIMILARITY_MIN", "0.65")))
    search_limit: int = field(default_factory=lambda: int(os.getenv("LEARNING_SEARCH_LIMIT", "3")))
    repeat_errors: int = field(default_factory=lambda: int(os.getenv("LEARNING_REPEAT_ERRORS", "2")))
    min_attempts: int = field(default_factory=lambda: int(os.getenv("LEARNING_MIN_ATTEMPTS", "3")))
    review_days: float = field(default_factory=lambda: float(os.getenv("LEARNING_REVIEW_DAYS", "3")))
    recent_window: int = field(default_factory=lambda: int(os.getenv("LEARNING_RECENT_WINDOW", "6")))
    silence_limit: float = field(default_factory=lambda: float(os.getenv("LEARNING_SILENCE_LIMIT", "95")))
    disagreement_limit: float = field(default_factory=lambda: float(os.getenv("LEARNING_DISAGREEMENT_LIMIT", "0")))
    role_timeout: float = field(default_factory=lambda: float(os.getenv("LEARNING_ROLE_TIMEOUT", "15")))
    max_model_calls: int = field(default_factory=lambda: int(os.getenv("LEARNING_MAX_MODEL_CALLS", "4")))
    max_tool_calls: int = field(default_factory=lambda: int(os.getenv("LEARNING_MAX_TOOL_CALLS", "4")))
    model: str = field(default_factory=lambda: os.getenv("LEARNING_MODEL", "gpt-4o-mini"))
    mode: str = field(default_factory=lambda: os.getenv("LEARNING_MODE", "full"))
    guide_path: Optional[str] = field(default_factory=lambda: os.getenv("LEARNING_GUIDE_PATH"))

    def __post_init__(self):
        if self.mode not in {"baseline", "history_coach", "full"}:
            raise ValueError("Invalid LEARNING_MODE")
        if min(self.repeat_errors, self.min_attempts, self.recent_window, self.max_model_calls, self.max_tool_calls, self.search_limit) < 1:
            raise ValueError("Learning limits must be positive")
        if not -1 <= self.similarity_min <= 1:
            raise ValueError("Invalid semantic relevance threshold")
        if self.review_days < 0 or self.role_timeout <= 0 or not 0 <= self.silence_limit <= 100 or not 0 <= self.disagreement_limit <= 1:
            raise ValueError("Invalid learning thresholds")


class InternalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Quality(InternalModel):
    status: Literal["sufficient", "needs_review", "unmeasurable"]
    reasons: List[str]


class Observation(InternalModel):
    record_id: str
    learner_key: str
    processed_at: str
    context: Dict[str, Any]
    reference_text: str
    stt_text: str
    acoustic_text: Optional[str]
    method: str
    fallback: bool
    words: List[Dict[str, Any]]
    insertions: List[Dict[str, Any]]
    targets: Dict[str, Dict[str, int]]
    pronunciation_score: float
    meaning_score: int
    voice: Dict[str, Any]
    feedback: Dict[str, str]
    quality: Quality
    versions: Dict[str, Any]
    telemetry: Dict[str, Any]


class CoachInput(InternalModel):
    words: List[Dict[str, Any]]
    insertions: List[Dict[str, Any]] = Field(default_factory=list)
    quality: Quality
    base_feedback: Dict[str, str]
    priority: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    versions: Dict[str, Any] = Field(default_factory=dict)
    reference_text: str = ""
    method: str = ""


class CoachOutput(InternalModel):
    focus: Optional[str] = None
    claim: Literal["practice", "repeat", "improved", "maintained", "uncertain"] = "practice"
    guide_id: Optional[str] = None
    history_feedback_id: Optional[str] = None
    direction: Literal["current", "repeat_guidance", "different_explanation", "confirm_improvement", "new_target"] = "current"


class Step(InternalModel):
    step: str = Field(min_length=1)
    assistantMessage: str = Field(min_length=1)
    userIntent: str = Field(min_length=1)


class Level(InternalModel):
    levelTitle: str = Field(min_length=1)
    levelDescription: str = Field(min_length=1)
    steps: List[Step] = Field(min_length=3, max_length=3)


class ScenarioData(InternalModel):
    scenarioContext: str
    goal: str
    levels: List[Level] = Field(min_length=3, max_length=3)


class DesignInput(InternalModel):
    context: str
    goal: str
    plan: Dict[str, Any]
    levels: Optional[List[Dict[str, Any]]] = None
    level_index: int = 0
    step_index: int = 0


class DesignOutput(InternalModel):
    target: Optional[str] = None
    reason: str = ""
    difficulty: Literal["default", "easy", "transfer", "review"]
    data: ScenarioData


class VerificationOutput(InternalModel):
    valid: bool
    reasons: List[str]


def scenario_core(data):
    """Validate known contract fields while retaining original opaque metadata elsewhere."""
    return {"scenarioContext":data["scenarioContext"], "goal":data["goal"], "levels":[
        {"levelTitle":level["levelTitle"], "levelDescription":level["levelDescription"],
         "steps":[{k:step[k] for k in ("step","assistantMessage","userIntent")} for step in level["steps"]]}
        for level in data["levels"]]}
