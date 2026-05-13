import json
import re
from openai import OpenAI
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

EVA_PARK_PRINCIPLES = """
[EVA Park 기반 시나리오 설계 원칙]

1. 기능적 대화 중심
- 사용자가 실제 생활에서 자주 겪는 대화 상황을 다룬다.
- 예: 병원 접수, 음식 주문, 약속 잡기, 도움 요청, 정보 묻기.

2. 개인적 관련성
- 사용자의 삶과 직접 관련 있는 주제여야 한다.
- 취미, 가족, 일상 경험, 관심사, 자주 가는 장소 등을 반영할 수 있다.

3. 의미 전달 중심
- 문법적으로 완벽한 문장보다 상대방이 의도를 이해할 수 있는지를 중시한다.
- 핵심 대상, 요청 내용, 수량, 장소, 시간, 감정 등이 전달되는지를 기준으로 한다.

4. 반복 연습과 단계적 난이도
- 처음에는 짧고 단순한 발화로 시작한다.
- 이후 조건 추가, 상대방 질문, 돌발 상황 등으로 점진적으로 난이도를 높인다.

5. 일반화 가능성
- 앱 안에서 연습한 표현이 실제 생활에서도 사용 가능해야 한다.
- 같은 목표를 다른 장소, 다른 상대, 다른 표현으로 확장할 수 있어야 한다.

6. 사회적 참여 포함
- 단순한 기능적 요청 외에도 자기소개, 감정 표현, 취미 공유 등 사회적 대화도 포함한다.
- 삶의 참여와 의사소통 자신감 회복을 목표로 한다.
"""


def generate_scenario_levels(scenario_context: str, goal: str) -> dict:
    system_prompt = f"""
너는 언어재활 시나리오 설계 보조자이자 성인 의사소통 훈련용 시나리오 설계 전문가임.

아래 EVA Park 기반 설계 원칙을 반드시 따라야 함:
{EVA_PARK_PRINCIPLES}

사용자가 입력한 시나리오 상황과 목적을 바탕으로
3단계 난이도의 훈련 시나리오를 생성해야 함.

규칙:
1. 총 3개 level을 만들어야 함.
2. 각 level마다 step 3개를 만들어야 함.
3. 총 9개의 step이 생성되어야 함.
4. 난이도는 쉬운 수준 -> 중간 수준 -> 어려운 수준 순으로 점진적으로 올라가야 함.
5. 각 level 내의 step 1, 2, 3은 반드시 실제 상황에서 일어나는 시간 순서(앞→뒤)를 따라야 함.
   예: 카페에서 "주문 확인" → "결제 방식 묻기" → "영수증/포인트 적립" 순서가 맞음.
   절대 안 되는 예: "음료 나왔습니다" 이후에 "준비 시간이 얼마나 걸리나요?"처럼 시간이 역행하는 step.
6. 같은 level의 step 3개를 순서대로 이어 읽으면 하나의 자연스러운 대화 장면이 완성되어야 함.
   각 step의 assistantMessage가 이전 step의 사용자 답변을 받아 이어지는 방식이어야 함.
7. 각 step의 assistantMessage는 실제 현장에서 상대방이 할 법한 자연스러운 구어체여야 함.
8. 난이도가 올라갈수록 돌발 상황, 추가 조건, 더 복잡한 질문 등을 포함해 실제 대화 압박을 높여야 함.
9. 반드시 JSON만 반환해야 함.
10. JSON 바깥의 설명 문장, 마크다운 코드블록, 주석은 절대 포함하지 말 것.

각 필드 작성 기준:
- levelTitle: 해당 레벨의 핵심 상황을 5~15자로 요약 (예: "병원 접수하기", "문진/기본 확인")
- levelDescription: EVA Park 설계 원칙(기능적 대화, 의미 전달, 반복 연습 등)을 반영하여 해당 레벨에서 연습할 내용을 1~2문장으로 설명
- step: 해당 스텝에서 사용자가 연습할 행동을 5~15자로 요약 (예: "인적 사항 말하기", "간단히 증상 설명하기")
- assistantMessage: 실제 상황에서 상대방(접수 직원, 의사 등)이 사용자에게 하는 말 (자연스러운 구어체)
- userIntent: 사용자가 이 step에서 전달해야 할 핵심 의미와 연습 포인트를 1문장으로 안내 (의미 전달 중심으로 작성)
"""

    user_prompt = f"""
시나리오 상황: {scenario_context}
사용자 목적: {goal}

반드시 아래 형식의 JSON 객체만 반환. 반드시 levels 배열에 3개의 level이 모두 포함되어야 함:
{{
  "scenarioContext": "{scenario_context}",
  "goal": "{goal}",
  "levels": [
    {{
      "levelTitle": "병원 접수하기",
      "levelDescription": "인적 사항과 기본 증상을 간단하게 말하는 연습입니다. 짧고 핵심적인 발화로 의미를 전달하는 데 집중합니다.",
      "steps": [
        {{
          "step": "인적 사항 말하기",
          "assistantMessage": "성함이랑 생년월일이 어떻게 되세요?",
          "userIntent": "이름과 생년월일을 정확하게 말해 의미를 전달해 보세요."
        }},
        {{
          "step": "간단히 증상 설명하기",
          "assistantMessage": "어디가 불편해서 오셨나요?",
          "userIntent": "불편한 증상을 한두 단어라도 말해 상대방이 이해할 수 있게 전달해 보세요."
        }},
        {{
          "step": "증상 정도 말하기",
          "assistantMessage": "증상이 얼마나 심하신가요?",
          "userIntent": "증상의 심각도를 간단한 표현으로 상대방에게 전달해 보세요."
        }}
      ]
    }},
    {{
      "levelTitle": "문진/기본 확인",
      "levelDescription": "증상의 시작 시점, 지속 기간, 추가 증상을 자세히 설명하며 의사소통 목표를 구체화합니다.",
      "steps": [
        {{
          "step": "증상 시작 시점 말하기",
          "assistantMessage": "증상이 언제부터 시작됐나요?",
          "userIntent": "증상이 시작된 날짜나 기간을 말해 상대방이 상황을 파악할 수 있게 해보세요."
        }},
        {{
          "step": "지속 기간 설명하기",
          "assistantMessage": "그 증상이 얼마나 지속됐나요?",
          "userIntent": "증상이 계속된 기간을 구체적으로 말해보세요."
        }},
        {{
          "step": "추가 증상 말하기",
          "assistantMessage": "다른 불편한 증상은 없으신가요?",
          "userIntent": "함께 나타난 다른 증상이 있다면 핵심 단어로라도 전달해 보세요."
        }}
      ]
    }},
    {{
      "levelTitle": "진료 상황에서 자세히 대답하기",
      "levelDescription": "의사의 복잡한 질문에 맞춰 증상의 양상, 악화 요인, 과거 병력을 대답하며 실제 대화 상황에 가까운 연습을 합니다.",
      "steps": [
        {{
          "step": "증상 양상 설명하기",
          "assistantMessage": "통증이 어떤 느낌인가요? 찌르는 듯한가요, 욱신거리나요?",
          "userIntent": "통증의 느낌이나 양상을 구체적인 표현으로 전달해 보세요."
        }},
        {{
          "step": "악화 요인 말하기",
          "assistantMessage": "어떨 때 증상이 더 심해지나요?",
          "userIntent": "증상을 악화시키는 상황이나 행동을 핵심 의미 중심으로 말해보세요."
        }},
        {{
          "step": "과거 병력 말하기",
          "assistantMessage": "이런 증상이 전에도 있었나요? 평소에 드시는 약이 있으신가요?",
          "userIntent": "관련 병력이나 복용 중인 약을 상대방이 이해할 수 있게 전달해 보세요."
        }}
      ]
    }}
  ]
}}

위 예시는 병원 접수 시나리오임. 실제로는 아래 입력된 시나리오 상황과 목적에 맞게 모든 내용을 새롭게 작성할 것.
EVA Park 설계 원칙(기능적 대화, 개인적 관련성, 의미 전달 중심, 단계적 난이도, 일반화 가능성, 사회적 참여)을 반드시 반영할 것:
- 시나리오 상황: {scenario_context}
- 사용자 목적: {goal}

[최우선 검토 사항]
각 level 내 step 1 → step 2 → step 3이 실제 상황의 시간 흐름과 일치하는지 반드시 확인할 것.
step들을 순서대로 이어 읽었을 때 자연스러운 대화 한 장면이 되어야 하며,
시간 역행(이미 완료된 단계를 뒤 step에서 다시 다루는 것)은 절대 허용되지 않음.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    text = response.choices[0].message.content.strip()
    print("LLM RAW RESPONSE:")
    print(repr(text))

    if not text:
        raise ValueError("모델이 빈 응답을 반환했음.")

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON 형태를 찾지 못했음. raw={text}")

    text = text[start:end+1]

    return json.loads(text)