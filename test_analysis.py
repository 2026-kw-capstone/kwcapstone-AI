"""
STT 텍스트를 직접 입력하여 발음 분석 결과를 확인하는 테스트 스크립트.
S3 업로드나 음성 파일 없이 바로 실행 가능.

실행 방법:
    python test_analysis.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from services.score_service import evaluate_reference_response, evaluate_scenario_response


# -------------------------------------------------------
# 출력 헬퍼
# -------------------------------------------------------
def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_word_analysis(word_analysis: list):
    grade_symbol = {"good": "✅", "warn": "⚠️", "error": "❌"}
    print(f"\n  {'글자':<6} {'STT':<6} {'score':<8} {'grade':<8} errorType")
    print("  " + "-" * 55)
    for item in word_analysis:
        sym = grade_symbol.get(item.get("grade", ""), "?")
        print(
            f"  {item.get('refChar', ''):<6} "
            f"{item.get('hypChar', ''):<6} "
            f"{item.get('score', 0):<8} "
            f"{sym} {item.get('grade', ''):<6} "
            f"{str(item.get('errorType', ''))}"
        )


def print_result(result: dict):
    print(f"\n  referenceText    : {result.get('referenceText')}")
    print(f"  sttText          : {result.get('sttText')}")
    print(f"  pronunciationScore : {result.get('pronunciationScore')}점")
    print(f"  meaningDeliveryScore : {result.get('meaningDeliveryScore')}점")
    print(f"  feedback         : {result.get('feedback')}")
    print_word_analysis(result.get("wordAnalysis", []))


# -------------------------------------------------------
# 1. 나만의 문장노트 (reference) 테스트
# -------------------------------------------------------
REFERENCE_CASES = [
    {
        "desc": "정확하게 발음한 경우",
        "referenceText": "물 좀 주세요",
        "sttText": "물 좀 주세요",
    },
    {
        "desc": "초성 오류 (주 → 부)",
        "referenceText": "물 좀 주세요",
        "sttText": "물 좀 부세요",
    },
    {
        "desc": "음절 누락",
        "referenceText": "물 좀 주세요",
        "sttText": "무 좀 주세요",
    },
    {
        "desc": "전체 오류",
        "referenceText": "문 열어 주세요",
        "sttText": "무 너러 주세요",
    },
    {
        "desc": "받침 오류 (약 → 야)",
        "referenceText": "약 주세요",
        "sttText": "야 주세요",
    },
]


def test_reference():
    print_section("나만의 문장노트 (reference) 테스트")
    for i, case in enumerate(REFERENCE_CASES, 1):
        print(f"\n[케이스 {i}] {case['desc']}")
        result = evaluate_reference_response(
            reference_text=case["referenceText"],
            stt_text=case["sttText"],
        )
        print_result(result)


# -------------------------------------------------------
# 2. 시나리오 발화 분석 테스트
# -------------------------------------------------------
SCENARIO_CASES = [
    {
        "desc": "이름/생년월일 말하기 - 정확",
        "assistantMessage": "성함이랑 생년월일이 어떻게 되세요?",
        "userIntent": "이름과 생년월일을 정확하게 말해 보세요.",
        "sttText": "김철수이고요 1990년 3월 5일입니다",
    },
    {
        "desc": "증상 설명 - 의도 전달 성공",
        "assistantMessage": "어디가 불편해서 오셨나요?",
        "userIntent": "불편한 증상을 짧게 설명해 보세요.",
        "sttText": "머리가 많이 아파서요",
    },
    {
        "desc": "증상 설명 - 의도 전달 부족",
        "assistantMessage": "어디가 불편해서 오셨나요?",
        "userIntent": "불편한 증상을 짧게 설명해 보세요.",
        "sttText": "아 그냥 좀",
    },
    {
        "desc": "증상 정도 말하기",
        "assistantMessage": "증상이 얼마나 심하신가요?",
        "userIntent": "해당 증상이 얼마나 심한지 간단히 말해보세요.",
        "sttText": "많이 심해요 어제부터 계속",
    },
]


def test_scenario():
    print_section("시나리오 발화 분석 테스트 (LLM 호출 포함)")
    for i, case in enumerate(SCENARIO_CASES, 1):
        print(f"\n[케이스 {i}] {case['desc']}")
        step_content = f"AI 질문: {case['assistantMessage']}\n사용자 연습 목표: {case['userIntent']}"
        result = evaluate_scenario_response(
            step_content=step_content,
            stt_text=case["sttText"],
        )
        print(f"  assistantMessage : {case['assistantMessage']}")
        print(f"  userIntent       : {case['userIntent']}")
        print_result(result)


# -------------------------------------------------------
# 실행
# -------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="발음 분석 테스트")
    parser.add_argument(
        "--mode",
        choices=["reference", "scenario", "all"],
        default="all",
        help="실행할 테스트 선택 (기본: all)",
    )
    args = parser.parse_args()

    if args.mode in ("reference", "all"):
        test_reference()

    if args.mode in ("scenario", "all"):
        test_scenario()

    print("\n\n✅ 테스트 완료")
