"""
발음 평가 서비스 단위 테스트
(services/score_service.py)

- 네트워크 불필요, API 키 불필요 (LLM 호출 함수는 별도 표시)
- 순수 Python 연산만 사용 → 빠르고 안정적
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.score_service import (
    attach_syllable_timestamps,
    build_alignment,
    build_overall_rule_feedback,
    build_rule_based_analysis,
    calculate_reference_scores,
    decompose_hangul,
    evaluate_reference_response,
    analyze_syllable_difference,
    split_korean_chars,
)


# ── 한글 분리 ──────────────────────────────────────────────────────────────────

class TestSplitKoreanChars:
    def test_basic(self):
        assert split_korean_chars("안녕") == ["안", "녕"]

    def test_with_spaces(self):
        result = split_korean_chars("안 녕")
        assert result == ["안", "녕"]

    def test_empty(self):
        assert split_korean_chars("") == []

    def test_mixed(self):
        result = split_korean_chars("안녕 hello")
        # 공백 제거 후 한글+영문 모두 포함
        assert "안" in result and "녕" in result


# ── 한글 자모 분해 ─────────────────────────────────────────────────────────────

class TestDecomposeHangul:
    @pytest.mark.parametrize("char,expected", [
        ("안", {"initial": "ㅇ", "medial": "ㅏ", "final": "ㄴ"}),
        ("녕", {"initial": "ㄴ", "medial": "ㅕ", "final": "ㅇ"}),
        ("하", {"initial": "ㅎ", "medial": "ㅏ", "final": ""}),
        ("세", {"initial": "ㅅ", "medial": "ㅔ", "final": ""}),
        ("요", {"initial": "ㅇ", "medial": "ㅛ", "final": ""}),
    ])
    def test_decompose(self, char, expected):
        assert decompose_hangul(char) == expected

    def test_no_jongsung(self):
        result = decompose_hangul("아")
        assert result["final"] == ""

    def test_with_jongsung(self):
        result = decompose_hangul("닭")
        assert result["final"] != ""

    @pytest.mark.parametrize("non_hangul", ["a", "1", " ", "!", "A"])
    def test_non_hangul_returns_none(self, non_hangul):
        assert decompose_hangul(non_hangul) is None

    def test_all_basic_syllables(self):
        for char in "가나다라마바사아자차카타파하":
            result = decompose_hangul(char)
            assert result is not None
            assert all(k in result for k in ("initial", "medial", "final"))


# ── 음절 차이 분석 ─────────────────────────────────────────────────────────────

class TestAnalyzeSyllableDifference:
    def test_identical(self):
        result = analyze_syllable_difference("안", "안")
        assert result["type"] == "equal"
        assert all(v == "same" for v in result["phonemeDiff"].values())

    def test_initial_different(self):
        # 가(ㄱ+ㅏ) vs 나(ㄴ+ㅏ): 초성만 다름
        result = analyze_syllable_difference("가", "나")
        assert result["phonemeDiff"]["initial"] == "different"
        assert result["phonemeDiff"]["medial"] == "same"
        assert result["phonemeDiff"]["final"] == "same"

    def test_medial_different(self):
        # 가(ㅏ) vs 고(ㅗ): 중성만 다름
        result = analyze_syllable_difference("가", "고")
        assert result["phonemeDiff"]["medial"] == "different"
        assert result["phonemeDiff"]["initial"] == "same"

    def test_final_different(self):
        # 말(ㄹ) vs 맛(ㅅ): 종성만 다름
        result = analyze_syllable_difference("말", "맛")
        assert result["phonemeDiff"]["final"] == "different"
        assert result["phonemeDiff"]["initial"] == "same"
        assert result["phonemeDiff"]["medial"] == "same"

    def test_all_different(self):
        result = analyze_syllable_difference("갈", "뉴")
        diff_count = sum(1 for v in result["phonemeDiff"].values() if v == "different")
        assert diff_count == 3

    def test_non_hangul_returns_unknown(self):
        result = analyze_syllable_difference("a", "b")
        assert result["type"] == "unknown"


# ── 점수 계산 ──────────────────────────────────────────────────────────────────

class TestCalculateReferenceScores:
    def test_perfect_match(self):
        scores = calculate_reference_scores("안녕하세요", "안녕하세요")
        assert scores["wer"] == 0.0
        assert scores["cer"] == 0.0
        assert scores["similarityScore"] == 100.0
        assert scores["pronunciationScore"] == 100.0

    def test_empty_reference(self):
        scores = calculate_reference_scores("", "안녕")
        assert scores["pronunciationScore"] == 0.0
        assert scores["similarityScore"] == 0.0

    def test_empty_stt(self):
        scores = calculate_reference_scores("안녕하세요", "")
        assert scores["pronunciationScore"] == 0.0

    def test_partial_match(self):
        scores = calculate_reference_scores("안녕하세요", "안녕")
        assert 0.0 < scores["similarityScore"] < 100.0
        assert scores["wer"] > 0.0

    def test_completely_different(self):
        scores = calculate_reference_scores("안녕하세요", "감사합니다")
        assert scores["similarityScore"] < 50.0

    def test_score_types(self):
        scores = calculate_reference_scores("안녕", "안녕")
        assert isinstance(scores["pronunciationScore"], float)
        assert isinstance(scores["similarityScore"], float)
        assert isinstance(scores["wer"], float)
        assert isinstance(scores["cer"], float)

    @pytest.mark.parametrize("ref,hyp", [
        ("물 좀 주세요", "물 좀 주세요"),
        ("약 주세요", "야 주세요"),
        ("문 열어 주세요", "무 너러 주세요"),
    ])
    def test_various_cases(self, ref, hyp):
        scores = calculate_reference_scores(ref, hyp)
        assert 0.0 <= scores["pronunciationScore"] <= 100.0
        assert 0.0 <= scores["similarityScore"] <= 100.0


# ── 정렬 분석 ──────────────────────────────────────────────────────────────────

class TestBuildAlignment:
    def test_perfect_match(self):
        result = build_alignment("안녕", "안녕")
        assert all(item["type"] == "equal" for item in result["aligned"])
        assert result["mismatchIndexes"] == []

    def test_result_keys(self):
        result = build_alignment("안녕하세요", "안녕하세요")
        assert set(result.keys()) == {"referenceChars", "sttChars", "aligned", "mismatchIndexes"}

    def test_substitution_detected(self):
        result = build_alignment("가나다", "가나라")
        types = [item["type"] for item in result["aligned"]]
        assert "substitute" in types

    def test_deletion_detected(self):
        # "가나다" → "가다" : "나" 누락
        result = build_alignment("가나다", "가다")
        types = [item["type"] for item in result["aligned"]]
        assert "delete" in types

    def test_insertion_detected(self):
        # "가다" → "가나다" : "나" 삽입
        result = build_alignment("가다", "가나다")
        types = [item["type"] for item in result["aligned"]]
        assert "insert" in types

    def test_mismatch_indexes_populated(self):
        result = build_alignment("가나다", "가라다")
        assert len(result["mismatchIndexes"]) > 0

    def test_reference_chars_order(self):
        result = build_alignment("안녕하세요", "안녕하세요")
        assert result["referenceChars"] == ["안", "녕", "하", "세", "요"]


# ── 규칙 기반 분석 ─────────────────────────────────────────────────────────────

class TestBuildRuleBasedAnalysis:
    def test_perfect_match_scores(self):
        alignment = build_alignment("안녕", "안녕")
        word_analysis, inserts, _ = build_rule_based_analysis("안녕", alignment)

        assert all(item["score"] == 100 for item in word_analysis)
        assert all(item["grade"] == "good" for item in word_analysis)

    def test_no_inserts_on_perfect(self):
        alignment = build_alignment("안녕", "안녕")
        _, inserts, _ = build_rule_based_analysis("안녕", alignment)
        assert inserts == []

    def test_delete_gives_zero_score(self):
        alignment = build_alignment("가나다", "가다")
        word_analysis, _, _ = build_rule_based_analysis("가나다", alignment)
        scores = [item["score"] for item in word_analysis]
        assert 0 in scores

    def test_score_rubric(self):
        """
        점수 루브릭:
          차이 0개 → 100, 1개 → 70, 2개 → 40, 3개 → 10, 누락 → 0
        """
        # 초성만 다른 대체: score=70 이어야 함
        alignment = build_alignment("가나다", "나나다")  # 가→나: 초성 ㄱ→ㄴ만 다름
        word_analysis, _, _ = build_rule_based_analysis("가나다", alignment)
        # 첫 번째 항목이 substitute
        first = next(
            (item for item in word_analysis if item.get("hypChar") == "나" and item.get("refChar") == "가"),
            None
        )
        if first:
            assert first["score"] == 70

    def test_grade_mapping(self):
        alignment = build_alignment("안녕하세요", "안녕하세요")
        word_analysis, _, _ = build_rule_based_analysis("안녕하세요", alignment)
        for item in word_analysis:
            assert item["grade"] in ("good", "warn", "error")
            assert item["score"] in (0, 10, 40, 70, 100)

    def test_word_analysis_item_keys(self):
        alignment = build_alignment("안녕", "안녕")
        word_analysis, _, _ = build_rule_based_analysis("안녕", alignment)
        required = {
            "refIndex", "refChar", "hypChar", "score", "grade",
            "errorType", "refParts", "hypParts", "phonemeDiff"
        }
        for item in word_analysis:
            assert required.issubset(item.keys())


# ── 전체 피드백 생성 ───────────────────────────────────────────────────────────

class TestBuildOverallRuleFeedback:
    def test_returns_string(self):
        alignment = build_alignment("안녕하세요", "안녕하세요")
        word_analysis, _, insert_feedbacks = build_rule_based_analysis("안녕하세요", alignment)
        feedback = build_overall_rule_feedback(word_analysis, insert_feedbacks)
        assert isinstance(feedback, str) and len(feedback) > 0

    def test_perfect_match_feedback(self):
        alignment = build_alignment("물 좀 주세요", "물 좀 주세요")
        word_analysis, _, insert_feedbacks = build_rule_based_analysis("물 좀 주세요", alignment)
        feedback = build_overall_rule_feedback(word_analysis, insert_feedbacks)
        # 오류 없으면 격려 피드백
        assert "잘" in feedback or "계속" in feedback

    def test_delete_error_feedback(self):
        alignment = build_alignment("가나다라", "나다라")
        word_analysis, _, insert_feedbacks = build_rule_based_analysis("가나다라", alignment)
        feedback = build_overall_rule_feedback(word_analysis, insert_feedbacks)
        assert isinstance(feedback, str) and len(feedback) > 0


# ── 타임스탬프 첨부 ────────────────────────────────────────────────────────────

class TestAttachSyllableTimestamps:
    def test_no_whisperx_sets_none(self):
        alignment = build_alignment("안녕", "안녕")
        word_analysis, _, _ = build_rule_based_analysis("안녕", alignment)
        result = attach_syllable_timestamps(word_analysis, "안녕", whisperx_words=None)

        for item in result:
            assert item["start"] is None
            assert item["end"] is None

    def test_with_whisperx_data(self):
        alignment = build_alignment("안녕하", "안녕하")
        word_analysis, _, _ = build_rule_based_analysis("안녕하", alignment)

        whisperx_words = [{"word": "안녕하", "start": 0.0, "end": 0.6}]
        result = attach_syllable_timestamps(word_analysis, "안녕하", whisperx_words=whisperx_words)

        for item in result:
            assert item["start"] is not None
            assert item["end"] is not None
            assert item["start"] >= 0.0


# ── evaluate_reference_response (통합) ────────────────────────────────────────

class TestEvaluateReferenceResponse:
    def test_result_keys(self):
        result = evaluate_reference_response("안녕하세요", "안녕하세요")
        required = ["referenceText", "sttText", "pronunciationScore", "feedback", "wordAnalysis"]
        for key in required:
            assert key in result, f"누락된 키: {key}"

    def test_removed_keys(self):
        """의미 전달률 및 내부 지표는 제거됨."""
        result = evaluate_reference_response("안녕하세요", "안녕하세요")
        for key in ("meaningDeliveryScore", "wer", "cer", "similarityScore",
                    "diffAnalysis", "insertions", "referenceSource", "evaluationMode"):
            assert key not in result, f"제거됐어야 할 키가 남아 있음: {key}"

    def test_perfect_score(self):
        result = evaluate_reference_response("안녕하세요", "안녕하세요")
        assert result["pronunciationScore"] == 100.0

    def test_score_range(self):
        result = evaluate_reference_response("안녕하세요", "감사합니다")
        assert 0.0 <= result["pronunciationScore"] <= 100.0

    def test_word_analysis_simplified(self):
        """wordAnalysis 항목은 refChar, hypChar, grade 만 포함."""
        result = evaluate_reference_response("물 좀 주세요", "물 좀 주세요")
        for item in result["wordAnalysis"]:
            assert set(item.keys()) == {"refChar", "hypChar", "grade"}
            assert item["grade"] in ("good", "warn", "error")

    def test_feedback_is_string(self):
        result = evaluate_reference_response("안녕하세요", "안녕하세요")
        assert isinstance(result["feedback"], str) and len(result["feedback"]) > 0

    @pytest.mark.parametrize("ref,hyp,desc", [
        ("물 좀 주세요",   "물 좀 주세요",   "정확한 발음"),
        ("물 좀 주세요",   "물 좀 부세요",   "초성 오류 (주→부)"),
        ("물 좀 주세요",   "무 좀 주세요",   "음절 누락"),
        ("문 열어 주세요", "무 너러 주세요", "전체 오류"),
        ("약 주세요",      "야 주세요",      "받침 오류 (약→야)"),
    ])
    def test_known_cases(self, ref, hyp, desc):
        result = evaluate_reference_response(ref, hyp)
        assert result["pronunciationScore"] >= 0
        if ref == hyp:
            assert result["pronunciationScore"] == 100.0, f"[{desc}] 완벽 매칭인데 100점이 아님"
        else:
            assert result["pronunciationScore"] < 100.0, f"[{desc}] 오류 있는데 100점"
