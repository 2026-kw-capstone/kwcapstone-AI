from openai import OpenAI
from config.settings import OPENAI_API_KEY
from typing import List, Dict, Optional

client = OpenAI(api_key=OPENAI_API_KEY)


def build_context_messages(
    chat_history: Optional[List[Dict[str, str]]] = None,
    max_pairs: int = 3
) -> List[Dict[str, str]]:
    if not chat_history:
        return []

    allowed_roles = {"user", "assistant"}

    normalized = []
    for item in chat_history:
        role = item.get("role", "").strip()
        content = item.get("content", "").strip()

        if role not in allowed_roles:
            continue
        if not content:
            continue

        normalized.append({"role": role, "content": content})

    pairs = []
    current_user = None

    for msg in normalized:
        if msg["role"] == "user":
            current_user = msg
        elif msg["role"] == "assistant" and current_user is not None:
            pairs.append([current_user, msg])
            current_user = None

    recent_pairs = pairs[-max_pairs:]

    context_messages = []
    for user_msg, assistant_msg in recent_pairs:
        context_messages.append(user_msg)
        context_messages.append(assistant_msg)

    return context_messages


def parse_reply_and_feedback(raw_text: str) -> Dict[str, str]:
    raw_text = raw_text.strip()

    reply_prefix = "답변:"
    feedback_prefix = "피드백:"

    reply = ""
    feedback = ""

    if reply_prefix in raw_text and feedback_prefix in raw_text:
        reply_start = raw_text.find(reply_prefix) + len(reply_prefix)
        feedback_start = raw_text.find(feedback_prefix)

        reply = raw_text[reply_start:feedback_start].strip()
        feedback = raw_text[feedback_start + len(feedback_prefix):].strip()
    else:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if len(lines) >= 2:
            reply = lines[0]
            feedback = " ".join(lines[1:])
        elif len(lines) == 1:
            reply = lines[0]
            feedback = "천천히 말해도 괜찮아요."
        else:
            reply = "천천히 다시 말해주셔도 괜찮아요."
            feedback = "좋아요. 부담 없이 이어가면 돼요."

    return {"reply": reply, "feedback": feedback}


def generate_free_talk_reply(
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, str]:
    system_prompt = """
너는 성인 언어장애/조음장애 사용자를 위한 따뜻한 대화 파트너임.

목표:
- 사용자가 편안하게 말하도록 돕기
- 짧고 쉬운 문장으로 답하기
- 실수보다 자신감을 살리는 방향으로 돕기
- 이전 대화 맥락이 있으면 자연스럽게 이어가기

반드시 아래 형식으로만 출력:
답변: <사용자에게 보여줄 따뜻한 답변 1~3문장>
피드백: <아주 짧은 긍정 피드백 1문장>

추가 규칙:
1. 답변은 너무 길지 않게 작성
2. 쉬운 문장 사용
3. 부담 주지 않기
4. 필요하면 천천히 다시 말해보라고 부드럽게 유도
5. 피드백은 짧고 긍정적으로 작성
6. 피드백에는 교정보다 잘한 점 중심으로 작성
7. 한 번에 질문을 너무 많이 하지 않기
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(build_context_messages(chat_history, max_pairs=3))
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        messages=messages
    )

    raw_content = response.choices[0].message.content.strip()
    parsed = parse_reply_and_feedback(raw_content)

    return {
        "reply": parsed["reply"],
        "feedback": parsed["feedback"],
        "assistant_message_for_history": parsed["reply"]
    }
