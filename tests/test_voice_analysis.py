"""
음성 품질 분석 서비스 단위 테스트
(services/voice_analysis_service.py)

- 네트워크 불필요, API 키 불필요
- conftest.py 의 합성 WAV 픽스처 사용
- 실제 librosa 연산으로 구조·등급·수치 검증
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.voice_analysis_service import (
    analyze_silence_ratio,
    analyze_speech_rate,
    analyze_voice,
    count_korean_syllables,
)


# ── 한국어 음절 수 계산 ────────────────────────────────────────────────────────

class TestCountKoreanSyllables:
    def test_pure_korean(self):
        assert count_korean_syllables("안녕하세요") == 5

    def test_mixed_text(self):
        # 한글만 카운트: 안녕(2) + 반갑습니다(5) = 7
        assert count_korean_syllables("안녕 hello 반갑습니다") == 7

    def test_empty(self):
        assert count_korean_syllables("") == 0

    def test_only_english(self):
        assert count_korean_syllables("hello world") == 0

    def test_numbers_and_symbols(self):
        assert count_korean_syllables("1234!@#") == 0

    @pytest.mark.parametrize("text,expected", [
        ("가", 1),
        ("가나다", 3),
        ("물 좀 주세요", 5),       # 물/좀/주/세/요
        ("진료 예약하러 왔어요", 9),
    ])
    def test_various_sentences(self, text, expected):
        assert count_korean_syllables(text) == expected


# ── 발화 속도 분석 ─────────────────────────────────────────────────────────────

class TestAnalyzeSpeechRate:
    def test_result_structure(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, "안녕하세요")

        assert set(result.keys()) == {"syllablesPerSecond", "score", "grade", "label"}

    def test_empty_text_returns_slow(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, "")

        assert result["grade"] == "slow"
        assert result["syllablesPerSecond"] == 0.0
        assert result["score"] == 0

    def test_non_korean_text_returns_slow(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, "hello world")

        assert result["grade"] == "slow"
        assert result["syllablesPerSecond"] == 0.0

    def test_grade_is_valid(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, "안녕하세요")

        assert result["grade"] in ("good", "slow", "fast")
        assert result["syllablesPerSecond"] >= 0

    def test_score_range(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, "안녕하세요")

        assert 0 <= result["score"] <= 100

    @pytest.mark.parametrize("text", [
        "가",
        "안녕하세요",
        "진료 예약을 하고 싶어요",
        "물 좀 주세요 감사합니다 안녕히 계세요",
    ])
    def test_various_texts(self, synthetic_wav, text):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_speech_rate(y, sr, text)

        assert result["grade"] in ("good", "slow", "fast")


# ── 침묵 비율 분석 ─────────────────────────────────────────────────────────────

class TestAnalyzeSilenceRatio:
    def test_result_structure(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_silence_ratio(y, sr)

        assert set(result.keys()) == {"pausePercent", "grade", "label"}

    def test_percent_range(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_silence_ratio(y, sr)

        assert 0.0 <= result["pausePercent"] <= 100.0

    def test_short_gap_not_counted(self):
        """300ms 신호 + 100ms 침묵 + 300ms 신호 → gap < 250ms → pausePercent=0."""
        sr = 16000
        voiced = np.sin(2 * np.pi * 440 * np.arange(int(0.3 * sr)) / sr).astype(np.float32) * 0.3
        silence = np.zeros(int(0.1 * sr), dtype=np.float32)
        y = np.concatenate([voiced, silence, voiced])
        result = analyze_silence_ratio(y, sr)
        assert result["pausePercent"] == 0.0

    def test_grade_is_valid(self, synthetic_wav):
        y, sr = librosa.load(synthetic_wav, sr=None, mono=True)
        result = analyze_silence_ratio(y, sr)

        assert result["grade"] in ("good", "warn", "error")

    def test_mostly_silent_triggers_warn_or_error(self, mostly_silent_wav):
        """침묵 ~80% → warn 또는 error."""
        y, sr = librosa.load(mostly_silent_wav, sr=None, mono=True)
        result = analyze_silence_ratio(y, sr)

        assert result["grade"] in ("warn", "error")
        assert result["pausePercent"] > 25

    def test_fully_silent_audio(self):
        """완전 무음 신호도 예외 없이 처리."""
        y = np.zeros(16000, dtype=np.float32)
        result = analyze_silence_ratio(y, 16000)

        assert set(result.keys()) == {"pausePercent", "grade", "label"}
        assert result["grade"] in ("good", "warn", "error")


# ── 통합: analyze_voice ────────────────────────────────────────────────────────

class TestAnalyzeVoice:
    def test_full_structure(self, synthetic_wav):
        result = analyze_voice(synthetic_wav, "안녕하세요")

        assert "speechRate" in result
        assert "silenceRatio" in result

    def test_all_subkeys(self, synthetic_wav):
        result = analyze_voice(synthetic_wav, "안녕하세요")
        assert set(result["speechRate"].keys()) == {"syllablesPerSecond", "score", "grade", "label"}
        assert set(result["silenceRatio"].keys()) == {"pausePercent", "grade", "label"}

    def test_all_grades_are_valid(self, synthetic_wav):
        result = analyze_voice(synthetic_wav, "안녕하세요")

        assert result["speechRate"]["grade"] in ("good", "slow", "fast")
        assert result["silenceRatio"]["grade"] in ("good", "warn", "error")

    def test_empty_stt_text(self, synthetic_wav):
        """STT 텍스트 없어도 예외 없이 동작해야 함."""
        result = analyze_voice(synthetic_wav, "")

        assert result["speechRate"]["grade"] == "slow"

    def test_mostly_silent_silence_ratio(self, mostly_silent_wav):
        result = analyze_voice(mostly_silent_wav, "가")
        assert result["silenceRatio"]["grade"] in ("warn", "error")
