"""Redis 기반 시나리오 훈련 세션 저장소.

Stateless였던 API를 agent 방식으로 바꾸면서, 매 턴마다 사용자의 진행 상황
(현재 레벨/스텝, 이전 발화·점수 히스토리, 난이도 결정 근거)을 서버가 기억해야 한다.
세션 1건을 JSON 문자열로 직렬화해 Redis 키 하나에 저장하고 TTL을 건다.
"""

import json
import uuid
from typing import Any, Dict, Optional

import redis

from config.settings import REDIS_URL, SESSION_TTL_SECONDS

_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def _key(session_id: str) -> str:
    return f"scenario:session:{session_id}"


def create_session(scenario_context: str, goal: str, max_steps: int = 9) -> Dict[str, Any]:
    """새 훈련 세션을 만들고 Redis에 저장한다. 아직 step은 비어 있다."""
    session_id = uuid.uuid4().hex
    session: Dict[str, Any] = {
        "sessionId": session_id,
        "scenarioContext": scenario_context,
        "goal": goal,
        "maxSteps": max_steps,
        "stepsCompleted": 0,
        "currentLevel": 1,
        "currentStepInLevel": 1,
        # 사용자가 지금 답해야 할 step (assistantMessage/userIntent 포함)
        "currentStep": None,
        # 완료된 턴들의 누적 기록 (난이도 결정의 근거가 됨)
        "history": [],
        "status": "active",  # active | completed
    }
    save_session(session)
    return session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    raw = _get_client().get(_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)


def save_session(session: Dict[str, Any]) -> None:
    """세션을 저장하고 TTL을 갱신한다."""
    _get_client().set(
        _key(session["sessionId"]),
        json.dumps(session, ensure_ascii=False),
        ex=SESSION_TTL_SECONDS,
    )


def delete_session(session_id: str) -> None:
    _get_client().delete(_key(session_id))
