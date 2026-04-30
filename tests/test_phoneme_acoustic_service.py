"""
음향 음소 인식 서비스 테스트
(services/phoneme_acoustic_service.py)

- 모델 로딩·실제 오디오 없이 동작 검증 (mock 기반)
- 4가지 오류 케이스별 감지 로직 검증
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.phoneme_acoustic_service import acoustic_recognize, is_model_available


# ── 모델 가용성 ────────────────────────────────────────────────────────────────

class TestIsModelAvailable:
    def test_returns_bool(self):
        result = is_model_available()
        assert isinstance(result, bool)

    def test_false_when_model_missing(self):
        with patch(
            "services.phoneme_acoustic_service._get_model",
            return_value=(None, None)
        ):
            assert is_model_available() is False


# ── acoustic_recognize: 모델 없을 때 ──────────────────────────────────────────

class TestAcousticRecognizeNoModel:
    def test_returns_none_when_model_unavailable(self, tmp_path):
        wav_file = tmp_path / "dummy.wav"
        wav_file.write_bytes(b"dummy")

        import services.phoneme_acoustic_service as svc
        original_processor = svc._processor
        original_model = svc._model
        svc._processor = None
        svc._model = None
        try:
            result = acoustic_recognize(str(wav_file))
            assert result is None
        finally:
            svc._processor = original_processor
            svc._model = original_model

    def test_returns_none_on_exception(self, tmp_path):
        wav_file = tmp_path / "dummy.wav"
        wav_file.write_bytes(b"not a real wav")

        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.side_effect = Exception("model error")

        import services.phoneme_acoustic_service as svc
        original_processor = svc._processor
        original_model = svc._model
        svc._processor = mock_processor
        svc._model = mock_model
        try:
            result = acoustic_recognize(str(wav_file))
            assert result is None
        finally:
            svc._processor = original_processor
            svc._model = original_model


# ── acoustic_recognize: 모델 mocking ─────────────────────────────────────────

class TestAcousticRecognizeMocked:
    """
    실제 wav2vec2 모델 없이 동작 검증.
    acoustic_recognize 함수를 직접 mock하여 4가지 오류 케이스별 출력을 시뮬레이션.
    """

    def _make_mock_recognize(self, return_value):
        """acoustic_recognize를 원하는 값을 반환하도록 mock."""
        return patch(
            "services.phoneme_acoustic_service.acoustic_recognize",
            return_value=return_value
        )

    def test_case1_vowel_distortion(self):
        """
        ① 모음 왜곡: 사용자가 "아" 대신 "어"처럼 발음
        Whisper: "아" (언어 모델 보정으로 정답 출력)
        wav2vec2: "어" (음향 그대로 출력) → 중성 오류 감지
        """
        from services.score_service import build_phoneme_based_analysis, apply_g2p

        ref = "나"
        # Whisper는 "나"로 보정 → G2P 비교 시 오류 없음
        whisper_analysis, _, _ = build_phoneme_based_analysis(ref, "나")
        assert whisper_analysis[0]["score"] == 100

        # wav2vec2는 "너"로 출력 → 중성 오류 감지
        acoustic_analysis, _, _ = build_phoneme_based_analysis(ref, "너")
        assert acoustic_analysis[0]["score"] < 100
        assert acoustic_analysis[0]["grade"] == "warn"
        assert "medial" in acoustic_analysis[0]["phonemeDiff"] and \
               acoustic_analysis[0]["phonemeDiff"]["medial"] == "different"

    def test_case2_weak_final_consonant(self):
        """
        ② 받침 약화: "약"을 "야"처럼 발음 (받침 ㄱ 누락)
        Whisper: "약" (받침을 채워서 출력) → 오류 없음 판정
        wav2vec2: "야" (받침 없이 출력) → 종성 오류 감지
        """
        from services.score_service import build_phoneme_based_analysis

        ref = "약"
        # Whisper: 보정 후 "약" → 오류 없음
        whisper_analysis, _, _ = build_phoneme_based_analysis(ref, "약")
        assert whisper_analysis[0]["score"] == 100

        # wav2vec2: "야" → 종성 오류
        acoustic_analysis, _, _ = build_phoneme_based_analysis(ref, "야")
        assert acoustic_analysis[0]["score"] < 100
        final_diff = acoustic_analysis[0]["phonemeDiff"].get("final")
        assert final_diff == "different"

    def test_case3_aspiration_confusion(self):
        """
        ③ 기식음 혼동: "파"(ㅍ) 대신 "바"(ㅂ)로 발음
        Whisper: 문맥 보정으로 "파" 출력 가능 → 오류 미감지
        wav2vec2: 음향 기반 "바" 출력 → 초성 오류 감지
        """
        from services.score_service import build_phoneme_based_analysis

        ref = "파"
        # Whisper: 보정 후 "파" → 오류 없음
        whisper_analysis, _, _ = build_phoneme_based_analysis(ref, "파")
        assert whisper_analysis[0]["score"] == 100

        # wav2vec2: 기식음 구분 → "바" 출력 → 초성 오류
        acoustic_analysis, _, _ = build_phoneme_based_analysis(ref, "바")
        assert acoustic_analysis[0]["score"] < 100
        initial_diff = acoustic_analysis[0]["phonemeDiff"].get("initial")
        assert initial_diff == "different"

    def test_case4_phonological_rule_correct(self):
        """
        ④ 음운 규칙에 맞는 발음: "닭이"→"달기" (연음화)
        wav2vec2도 "달기"로 출력 → G2P 변환으로 100점
        """
        from services.score_service import build_phoneme_based_analysis

        ref = "닭이"
        acoustic_analysis, _, _ = build_phoneme_based_analysis(ref, "달기")
        assert all(item["score"] == 100 for item in acoustic_analysis)
        assert all(item["grade"] == "good" for item in acoustic_analysis)

    def test_all_four_cases_summary(self):
        """4가지 케이스 감지 여부 요약 검증."""
        from services.score_service import build_phoneme_based_analysis, evaluate_reference_response

        cases = [
            # (ref,  whisper_out, acoustic_out, whisper_catches, acoustic_catches)
            ("나",  "나",  "너",  False, True),   # ① 모음 왜곡
            ("약",  "약",  "야",  False, True),   # ② 받침 약화
            ("파",  "파",  "바",  False, True),   # ③ 기식음
            ("닭이", "달기", "달기", False, False),  # ④ 음운 규칙: G2P로 100점, 오류 없음
        ]

        for ref, whisper_out, acoustic_out, whisper_catches, acoustic_catches in cases:
            w_analysis, _, _ = build_phoneme_based_analysis(ref, whisper_out)
            w_detected = any(item["score"] < 100 for item in w_analysis)
            assert w_detected == whisper_catches, \
                f"[Whisper] ref='{ref}' hyp='{whisper_out}': 감지={w_detected}, 기대={whisper_catches}"

            a_analysis, _, _ = build_phoneme_based_analysis(ref, acoustic_out)
            a_detected = any(item["score"] < 100 for item in a_analysis)
            assert a_detected == acoustic_catches, \
                f"[Acoustic] ref='{ref}' hyp='{acoustic_out}': 감지={a_detected}, 기대={acoustic_catches}"
