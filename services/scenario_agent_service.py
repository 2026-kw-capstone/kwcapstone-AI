"""시나리오 훈련을 LLM Agent 방식으로 구동하는 서비스.

기존 `scenario_service.generate_scenario_levels`는 9개 step을 '한 번에' 생성하는
파이프라인이었다. 그래서 사용자가 각 step에서 실제로 어떻게 말했는지를 반영해
다음 step을 조정할 수 없었다.

이 모듈은 그 흐름을 agent로 바꾼다:
- LLM이 도구(tool)를 직접 호출하며 매 턴마다 다음 행동을 결정한다.
- 사용자 발화를 채점(evaluate_response)하고, 점수를 보고 난이도를 정한 뒤
  (generate_next_step), 마지막 step이면 세션을 종료(finish_session)한다.
- step은 미리 만들어두지 않고, 직전 사용자 발화를 받은 다음 동적으로 생성한다.

채점 로직(score_service)은 별도 API가 아니라 agent의 도구로 통합되어 있다.
"""

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config.settings import OPENAI_API_KEY
from services.scenario_service import EVA_PARK_PRINCIPLES, _parse_llm_json
from services.score_service import evaluate_scenario_response

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o"

# 난이도 기준(나아갈 레벨의 baseline). agent가 점수를 보고 이 안에서 조정한다.
_DIFFICULTY_GUIDE = {
    "easy": "단답형 또는 1~2어절로 답할 수 있는 단순 요청/선택. 질문이 명확하고 예측 가능.",
    "normal": "2~3가지 정보를 함께 전달해야 하는 상황. 상대가 되묻거나 추가 설명을 요구할 수 있음.",
    "hard": "돌발 상황(품절·변경 등)·여러 조건 동시 처리·의견/감정을 포함한 긴 발화가 필요. 대화 압박이 큼.",
}


# ── 단일 step 동적 생성 (LLM sub-call) ──────────────────────────────────────
def generate_next_step_llm(
    scenario_context: str,
    goal: str,
    difficulty: str,
    history: List[Dict[str, Any]],
    nominal_level: int,
    step_in_level: int,
) -> Dict[str, str]:
    """직전까지의 대화 흐름을 받아 '다음 한 개' step만 생성한다."""
    difficulty = difficulty if difficulty in _DIFFICULTY_GUIDE else "normal"

    # 지금까지 완료된 대화 흐름을 줄글로 정리
    if history:
        flow_lines = []
        for i, turn in enumerate(history, start=1):
            step = turn.get("step", {})
            flow_lines.append(
                f"{i}) 상대 발화: {step.get('assistantMessage', '')}\n"
                f"   사용자 발화: {turn.get('userUtterance', '')}\n"
                f"   의미전달 점수: {turn.get('meaningDeliveryScore', 'N/A')} / "
                f"발음 점수: {turn.get('pronunciationScore', 'N/A')}"
            )
        flow_text = "\n".join(flow_lines)
    else:
        flow_text = "(아직 진행된 대화 없음 — 이번이 첫 step)"

    system_prompt = f"""
너는 언어재활 시나리오 설계 보조자이자 성인 의사소통 훈련용 시나리오 설계 전문가임.

아래 EVA Park 기반 설계 원칙을 반드시 따라야 함:
{EVA_PARK_PRINCIPLES}

너는 전체 시나리오를 미리 다 만드는 게 아니라, 지금까지의 실제 대화 흐름을 보고
'다음 한 개의 step'만 생성한다.

규칙:
1. 지금까지의 대화와 자연스럽게 이어지는 step이어야 함 (시간 역행 금지).
2. assistantMessage는 반드시 상대방(직원·의사 등) 역할의 발화이며, 사용자 발화가 아님.
3. assistantMessage는 yes/no로만 답할 수 있는 질문이어서는 절대 안 됨.
   사용자가 구체적 내용을 발화해야 하는 개방형 질문/요청이어야 함.
4. 이번 step의 난이도는 '{difficulty}' 수준이어야 함:
   {_DIFFICULTY_GUIDE[difficulty]}
5. 반드시 JSON만 반환. 마크다운 코드블록·주석·설명 문장 절대 포함 금지.
"""

    user_prompt = f"""
시나리오 상황: {scenario_context}
사용자 목적: {goal}
현재 진행: 레벨 {nominal_level}, 레벨 내 {step_in_level}번째 step
이번 step 목표 난이도: {difficulty}

[지금까지의 대화 흐름]
{flow_text}

반드시 아래 JSON 형식만 반환:
{{
  "step": "사용자가 연습할 행동 5~15자 요약",
  "assistantMessage": "상대방이 사용자에게 먼저 건네는 자연스러운 구어체 발화",
  "userIntent": "사용자가 전달해야 할 핵심 의미와 연습 포인트 1문장"
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content
    parsed = _parse_llm_json(raw)
    return {
        "step": parsed.get("step", "").strip(),
        "assistantMessage": parsed.get("assistantMessage", "").strip(),
        "userIntent": parsed.get("userIntent", "").strip(),
    }


# ── Agent 도구 정의 ─────────────────────────────────────────────────────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_response",
            "description": (
                "직전 사용자 음성 발화를 현재 step 맥락에서 채점한다. "
                "발음 점수·의미전달 점수·피드백을 반환한다. "
                "다음 난이도를 정하기 전에 반드시 먼저 호출해야 한다."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_next_step",
            "description": (
                "다음 훈련 step을 생성한다. 직전 점수가 낮으면 난이도를 낮추고(easy), "
                "높으면 올린다(hard). 점수를 근거로 difficulty를 결정해야 한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "normal", "hard"],
                        "description": "이번 step 난이도. 직전 점수 기반으로 결정.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "이 난이도를 고른 근거 1문장 (점수 인용).",
                    },
                },
                "required": ["difficulty", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_session",
            "description": "정해진 step 수를 모두 마쳤을 때 세션을 종료한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "closingMessage": {
                        "type": "string",
                        "description": "사용자에게 전할 따뜻한 마무리 인사 1~2문장.",
                    }
                },
                "required": ["closingMessage"],
            },
        },
    },
]


# ── 도구 실행(dispatch) ─────────────────────────────────────────────────────
def _nominal_position(steps_completed: int) -> Dict[str, int]:
    """완료한 step 수로 레벨/레벨내 위치를 계산 (3 step마다 레벨 상승)."""
    level = steps_completed // 3 + 1
    step_in_level = steps_completed % 3 + 1
    return {"level": min(level, 3), "stepInLevel": step_in_level}


def _dispatch_tool(
    name: str,
    args: Dict[str, Any],
    session: Dict[str, Any],
    turn_context: Dict[str, Any],
) -> Dict[str, Any]:
    """agent가 고른 도구를 실제로 실행하고 결과(dict)를 돌려준다.

    session/turn_context를 직접 변형(mutate)한다.
    """
    if name == "evaluate_response":
        current_step = session.get("currentStep") or {}
        step_content = (
            f"AI 질문: {current_step.get('assistantMessage', '')}\n"
            f"사용자 연습 목표: {current_step.get('userIntent', '')}"
        )
        eval_result = evaluate_scenario_response(
            step_content,
            turn_context["sttText"],
            audio_path=turn_context.get("audioPath"),
            voice_result=turn_context.get("voiceResult"),
        )
        turn_context["lastEvaluation"] = eval_result
        # agent에는 점수 위주로 요약 전달 (장황한 wordAnalysis는 생략)
        return {
            "pronunciationScore": eval_result.get("pronunciationScore"),
            "meaningDeliveryScore": eval_result.get("meaningDeliveryScore"),
            "pronunciationFeedback": eval_result.get("pronunciationFeedback"),
            "meaningDeliveryFeedback": eval_result.get("meaningDeliveryFeedback"),
        }

    if name == "generate_next_step":
        pos = _nominal_position(session["stepsCompleted"])
        step = generate_next_step_llm(
            scenario_context=session["scenarioContext"],
            goal=session["goal"],
            difficulty=args.get("difficulty", "normal"),
            history=session["history"],
            nominal_level=pos["level"],
            step_in_level=pos["stepInLevel"],
        )
        session["currentStep"] = step
        session["currentLevel"] = pos["level"]
        session["currentStepInLevel"] = pos["stepInLevel"]
        turn_context["nextStep"] = step
        turn_context["difficulty"] = args.get("difficulty", "normal")
        turn_context["difficultyReason"] = args.get("reason", "")
        return {"step": step, "difficulty": args.get("difficulty"), "accepted": True}

    if name == "finish_session":
        session["status"] = "completed"
        session["currentStep"] = None
        turn_context["closingMessage"] = args.get("closingMessage", "")
        turn_context["finished"] = True
        return {"finished": True}

    return {"error": f"unknown tool: {name}"}


# ── Agent 루프 ──────────────────────────────────────────────────────────────
def _run_agent(
    session: Dict[str, Any],
    instruction: str,
    turn_context: Dict[str, Any],
    max_iters: int = 6,
) -> None:
    """system 지시 + 현재 상황을 주고, agent가 도구를 호출하며 한 턴을 처리하게 한다."""
    system_prompt = f"""
너는 성인 언어재활 사용자를 위한 1:1 시나리오 훈련 코치 agent임.
사용자는 음성으로 답하고, 너는 매 턴마다 도구를 호출해 훈련을 진행한다.

진행 원칙:
- 사용자가 방금 답했다면 먼저 evaluate_response로 채점한다.
- 점수를 보고 다음 step 난이도를 정한다:
  · 의미전달/발음 점수가 낮으면(대략 70 미만) 난이도를 낮춰(easy) 자신감을 살린다.
  · 점수가 높으면(대략 85 이상) 난이도를 올려(hard) 도전하게 한다.
  · 그 사이면 normal로 자연스럽게 진행한다.
- 아직 목표 step 수({session['maxSteps']}개)에 도달하지 않았으면 generate_next_step으로
  다음 step을 만든다.
- 목표 step 수를 모두 마쳤으면 generate_next_step 대신 finish_session으로 종료한다.
- 한 턴에는 필요한 도구만 호출하고, 다음 step 생성(또는 종료)까지 끝내면 멈춘다.

현재 세션 상태:
- 시나리오 상황: {session['scenarioContext']}
- 사용자 목적: {session['goal']}
- 지금까지 완료한 step 수: {session['stepsCompleted']} / {session['maxSteps']}
"""

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # 더 호출할 도구가 없으면 종료
            break

        # assistant 메시지(도구 호출 포함)를 히스토리에 추가
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _dispatch_tool(tc.function.name, args, session, turn_context)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        # 이번 턴의 목표(다음 step 생성 또는 종료)가 끝났으면 루프 중단
        if turn_context.get("nextStep") or turn_context.get("finished"):
            break


# ── 외부 진입점 ─────────────────────────────────────────────────────────────
def start_session_turn(session: Dict[str, Any]) -> Dict[str, Any]:
    """세션 시작: 첫 step을 생성한다 (채점 없음)."""
    turn_context: Dict[str, Any] = {}
    instruction = (
        "훈련을 시작한다. 아직 사용자 발화가 없으니 채점은 하지 말고, "
        "generate_next_step으로 난이도 easy의 첫 step을 만들어라."
    )
    _run_agent(session, instruction, turn_context)
    return turn_context


def advance_session_turn(
    session: Dict[str, Any],
    stt_text: str,
    audio_path: Optional[str],
    voice_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """사용자 발화 한 턴 처리: 채점 → 난이도 결정 → 다음 step 또는 종료."""
    turn_context: Dict[str, Any] = {
        "sttText": stt_text,
        "audioPath": audio_path,
        "voiceResult": voice_result,
    }
    completed_step = session.get("currentStep")

    # 이번 답변까지 포함하면 몇 개째 step을 마치는지(off-by-one 방지).
    answered_count = session["stepsCompleted"] + 1
    is_last = answered_count >= session["maxSteps"]
    closing_clause = (
        f"이 답변으로 마지막 {session['maxSteps']}번째 step까지 모두 마쳤다. "
        "따라서 generate_next_step을 호출하지 말고 반드시 finish_session으로 종료하라."
        if is_last
        else "채점 후 generate_next_step으로 다음 step을 만들어라."
    )

    instruction = (
        f'사용자가 방금 음성으로 답했고, STT 결과는 다음과 같다: "{stt_text}"\n'
        f"이 답변까지 포함하면 {answered_count}/{session['maxSteps']} step을 마치는 것이다.\n"
        f"먼저 evaluate_response로 채점하고, 점수를 보고 난이도를 판단하라. {closing_clause}"
    )
    _run_agent(session, instruction, turn_context)

    # 이번 턴 기록을 히스토리에 누적 (다음 step 생성 시 흐름 근거가 됨)
    evaluation = turn_context.get("lastEvaluation", {})
    session["history"].append(
        {
            "step": completed_step,
            "userUtterance": stt_text,
            "pronunciationScore": evaluation.get("pronunciationScore"),
            "meaningDeliveryScore": evaluation.get("meaningDeliveryScore"),
            "difficulty": turn_context.get("difficulty"),
            "difficultyReason": turn_context.get("difficultyReason"),
        }
    )
    session["stepsCompleted"] += 1

    return turn_context
