import json
import re
from openai import OpenAI
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_scenario_levels(scenario_context: str, goal: str) -> dict:
    system_prompt = """
너는 성인 의사소통 훈련용 시나리오 설계 전문가임.

사용자가 입력한 시나리오 상황과 목적을 바탕으로
3단계 난이도의 훈련 시나리오를 생성해야 함.

규칙:
1. 총 3개 level을 만들어야 함.
2. 각 level마다 step 3개를 만들어야 함.
3. 총 9개의 step이 생성되어야 함.
4. 난이도는 쉬운 수준 -> 중간 수준 -> 어려운 수준 순으로 점진적으로 올라가야 함.
5. 각 Step은 실제 대화 흐름처럼 자연스럽게 이어져야 함.
6. 반드시 JSON만 반환해야 함.
7. JSON 바깥의 설명 문장, 마크다운 코드블록, 주석은 절대 포함하지 말 것.

각 필드 작성 기준:
- levelTitle: 해당 레벨의 핵심 상황을 5~15자로 요약 (예: "병원 접수하기", "문진/기본 확인")
- levelDescription: 해당 레벨에서 연습할 내용을 1~2문장으로 설명
- step: 해당 스텝에서 사용자가 연습할 행동을 5~15자로 요약 (예: "인적 사항 말하기", "간단히 증상 설명하기")
- assistantMessage: 실제 상황에서 상대방(접수 직원, 의사 등)이 사용자에게 하는 말 (자연스러운 구어체)
- userIntent: 사용자가 이 step에서 말해야 할 내용을 안내하는 연습 포인트 1문장
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
      "levelDescription": "인적 사항과 기본 증상을 간단하게 말하는 연습입니다.",
      "steps": [
        {{
          "step": "인적 사항 말하기",
          "assistantMessage": "성함이랑 생년월일이 어떻게 되세요?",
          "userIntent": "이름과 생년월일을 정확하게 말해 보세요."
        }},
        {{
          "step": "간단히 증상 설명하기",
          "assistantMessage": "어디가 불편해서 오셨나요?",
          "userIntent": "불편한 증상을 짧게 설명해 보세요."
        }},
        {{
          "step": "증상 정도 말하기",
          "assistantMessage": "증상이 얼마나 심하신가요?",
          "userIntent": "해당 증상이 얼마나 심한지 간단히 말해보세요."
        }}
      ]
    }},
    {{
      "levelTitle": "문진/기본 확인",
      "levelDescription": "증상의 시작 시점, 지속 기간, 추가 증상을 자세히 설명합니다.",
      "steps": [
        {{
          "step": "증상 시작 시점 말하기",
          "assistantMessage": "증상이 언제부터 시작됐나요?",
          "userIntent": "증상이 시작된 날짜나 기간을 말해보세요."
        }},
        {{
          "step": "지속 기간 설명하기",
          "assistantMessage": "그 증상이 얼마나 지속됐나요?",
          "userIntent": "증상이 계속된 기간을 구체적으로 말해보세요."
        }},
        {{
          "step": "추가 증상 말하기",
          "assistantMessage": "다른 불편한 증상은 없으신가요?",
          "userIntent": "함께 나타난 다른 증상이 있다면 말해보세요."
        }}
      ]
    }},
    {{
      "levelTitle": "진료 상황에서 자세히 대답하기",
      "levelDescription": "의사의 질문에 맞춰 증상의 양상과 악화 요인 등을 대답합니다.",
      "steps": [
        {{
          "step": "증상 양상 설명하기",
          "assistantMessage": "통증이 어떤 느낌인가요? 찌르는 듯한가요, 욱신거리나요?",
          "userIntent": "통증의 느낌이나 양상을 구체적으로 표현해보세요."
        }},
        {{
          "step": "악화 요인 말하기",
          "assistantMessage": "어떨 때 증상이 더 심해지나요?",
          "userIntent": "증상을 악화시키는 상황이나 행동을 말해보세요."
        }},
        {{
          "step": "과거 병력 말하기",
          "assistantMessage": "이런 증상이 전에도 있었나요? 평소에 드시는 약이 있으신가요?",
          "userIntent": "관련 병력이나 복용 중인 약을 말해보세요."
        }}
      ]
    }}
  ]
}}

위 예시는 병원 접수 시나리오임. 실제로는 아래 입력된 시나리오 상황과 목적에 맞게 모든 내용을 새롭게 작성할 것:
- 시나리오 상황: {scenario_context}
- 사용자 목적: {goal}
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
