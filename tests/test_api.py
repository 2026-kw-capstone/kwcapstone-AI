"""
전체 API 엔드포인트 통합 테스트
(api/app.py — FastAPI TestClient 사용)

기본 동작:
  - S3 다운로드    → 로컬 WAV 파일 복사로 대체 (mock)
  - Whisper STT   → 미리 정의한 텍스트 반환 (mock)
  - OpenAI 호출   → 미리 정의한 응답 반환 (mock)
  실제 외부 서비스를 호출하는 테스트는 환경 변수로 활성화:
    TEST_USE_REAL_WHISPER=1   Whisper 실제 실행 (tests/audio/speech.wav 필요)
    TEST_USE_REAL_OPENAI=1    OpenAI API 실제 호출 (OPENAI_API_KEY 필요)

실행 예시:
  pytest tests/test_api.py -v
  TEST_USE_REAL_OPENAI=1 pytest tests/test_api.py::TestFreeTalkEndpoint::test_free_talk_real_openai -v
"""

import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import make_s3_mock  # 로컬 파일 복사 mock helper

USE_REAL_OPENAI = os.getenv("TEST_USE_REAL_OPENAI", "0") == "1"
USE_REAL_WHISPER = os.getenv("TEST_USE_REAL_WHISPER", "0") == "1"


def _mock_preprocess(input_path: str, output_path: str) -> str:
    """pydub(ffprobe 필요) 대신 파일을 그대로 복사. 합성 WAV는 이미 mono 16kHz."""
    shutil.copy2(input_path, output_path)
    return output_path

# 기본 mock 값 ─────────────────────────────────────────────────────────────────
MOCK_STT = "안녕하세요 반갑습니다"
MOCK_MP3 = b"\xFF\xFB\x90\x00" * 512  # 가짜 MP3 헤더

MOCK_SCENARIO = {
    "scenarioContext": "병원 접수",
    "goal": "진료 예약하기",
    "levels": [
        {
            "levelTitle": "쉬운 병원 접수",
            "levelDescription": "간단한 표현으로 접수를 연습합니다.",
            "steps": [
                {"step": "인사하기",    "assistantMessage": "안녕하세요.",          "userIntent": "인사해보세요."},
                {"step": "용무 말하기", "assistantMessage": "어떻게 오셨나요?",     "userIntent": "진료 보러 왔다고 말해보세요."},
                {"step": "이름 말하기", "assistantMessage": "성함이 어떻게 되세요?", "userIntent": "이름을 말해보세요."},
            ],
        },
        {
            "levelTitle": "보통 병원 접수",
            "levelDescription": "증상을 설명하는 연습을 합니다.",
            "steps": [
                {"step": "증상 말하기", "assistantMessage": "어디가 불편하세요?",    "userIntent": "아픈 곳을 말해보세요."},
                {"step": "기간 말하기", "assistantMessage": "언제부터 아프셨나요?",  "userIntent": "아픈 기간을 말해보세요."},
                {"step": "예약 요청",   "assistantMessage": "예약 원하시나요?",      "userIntent": "예약하고 싶다고 말해보세요."},
            ],
        },
        {
            "levelTitle": "어려운 병원 접수",
            "levelDescription": "상세한 증상을 설명하는 연습을 합니다.",
            "steps": [
                {"step": "상세 증상", "assistantMessage": "자세히 말씀해주세요.",    "userIntent": "자세한 증상을 말해보세요."},
                {"step": "병력 설명", "assistantMessage": "기존 지병이 있으신가요?", "userIntent": "병력을 말해보세요."},
                {"step": "마무리",   "assistantMessage": "접수 완료됩니다.",         "userIntent": "감사하다고 말해보세요."},
            ],
        },
    ],
}

MOCK_SCENARIO_EVAL = {
    "inferredReferenceText": "진료 예약을 하러 왔어요",
    "meaningDeliveryScore": 85,
    "feedback": "자연스럽게 잘 전달했어요.",
}

MOCK_FREE_TALK = {
    "reply": "정말 그렇네요! 맑은 날씨가 기분을 좋게 해요.",
    "feedback": "자연스럽게 잘 표현했어요!",
    "assistant_message_for_history": "정말 그렇네요! 맑은 날씨가 기분을 좋게 해요.",
}

# TestClient 생성 ───────────────────────────────────────────────────────────────
from api.app import app

client = TestClient(app)


# ── 인증 비활성화 픽스처 ──────────────────────────────────────────────────────────
# COLAB_API_TOKEN 환경 변수가 설정되어 있어도 테스트에서는 인증을 건너뜀

@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    """모든 API 테스트에서 토큰 인증을 비활성화."""
    monkeypatch.setattr("api.app.COLAB_API_TOKEN", None)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_ok(self):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}

    def test_method_not_allowed(self):
        res = client.post("/health")
        assert res.status_code == 405


# ── /stt ──────────────────────────────────────────────────────────────────────

class TestSTTEndpoint:
    def test_response_structure(self, synthetic_wav):
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value=MOCK_STT):
            res = client.post("/stt", json={"s3Url": "https://mock.s3/audio.wav"})

        assert res.status_code == 200
        data = res.json()
        assert set(data.keys()) == {"success", "sttText"}

    def test_success_flag(self, synthetic_wav):
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value=MOCK_STT):
            res = client.post("/stt", json={"s3Url": "https://mock.s3/audio.wav"})

        assert res.json()["success"] is True

    def test_stt_text_returned(self, synthetic_wav):
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value=MOCK_STT):
            res = client.post("/stt", json={"s3Url": "https://mock.s3/audio.wav"})

        assert res.json()["sttText"] == MOCK_STT

    def test_s3_url_required(self):
        res = client.post("/stt", json={})
        assert res.status_code == 422

    @pytest.mark.skipif(not USE_REAL_WHISPER, reason="TEST_USE_REAL_WHISPER=1 로 활성화")
    def test_real_whisper(self, real_speech_wav, synthetic_wav):
        """실제 Whisper 모델 실행 (느림, 선택 사항)."""
        audio = real_speech_wav or synthetic_wav
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(audio)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess):
            res = client.post("/stt", json={"s3Url": "https://mock.s3/audio.wav"})

        assert res.status_code == 200
        assert res.json()["success"] is True
        assert isinstance(res.json()["sttText"], str)


# ── /tts ──────────────────────────────────────────────────────────────────────

class TestTTSEndpoint:
    def test_returns_mp3(self):
        with patch("api.app.text_to_speech", return_value=MOCK_MP3):
            res = client.post("/tts", json={"text": "안녕하세요"})

        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/mpeg"
        assert len(res.content) > 0

    def test_default_params(self):
        with patch("api.app.text_to_speech", return_value=MOCK_MP3) as mock_tts:
            client.post("/tts", json={"text": "테스트"})
            args, kwargs = mock_tts.call_args
            # 기본값: voice="nova", speed=1.0
            assert args[1] == "nova"
            assert args[2] == 1.0

    @pytest.mark.parametrize("voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    def test_all_voices(self, voice):
        with patch("api.app.text_to_speech", return_value=MOCK_MP3):
            res = client.post("/tts", json={"text": "안녕", "voice": voice})
        assert res.status_code == 200

    @pytest.mark.parametrize("speed", [0.25, 0.5, 1.0, 2.0, 4.0])
    def test_speed_range(self, speed):
        with patch("api.app.text_to_speech", return_value=MOCK_MP3):
            res = client.post("/tts", json={"text": "안녕", "speed": speed})
        assert res.status_code == 200

    def test_text_required(self):
        res = client.post("/tts", json={})
        assert res.status_code == 422

    @pytest.mark.skipif(not USE_REAL_OPENAI, reason="TEST_USE_REAL_OPENAI=1 로 활성화")
    def test_real_openai(self):
        """실제 OpenAI TTS 호출 (OPENAI_API_KEY 필요)."""
        res = client.post("/tts", json={"text": "안녕하세요", "voice": "nova", "speed": 1.0})
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/mpeg"
        assert len(res.content) > 100  # 실제 MP3는 충분히 큼


# ── /generate-scenario ────────────────────────────────────────────────────────

class TestGenerateScenarioEndpoint:
    def test_success_flag(self):
        with patch("api.app.generate_scenario_levels", return_value=MOCK_SCENARIO):
            res = client.post(
                "/generate-scenario",
                json={"scenarioContext": "병원 접수", "goal": "진료 예약하기"},
            )
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_data_key_present(self):
        with patch("api.app.generate_scenario_levels", return_value=MOCK_SCENARIO):
            res = client.post(
                "/generate-scenario",
                json={"scenarioContext": "병원 접수", "goal": "진료 예약하기"},
            )
        assert "data" in res.json()

    def test_three_levels(self):
        with patch("api.app.generate_scenario_levels", return_value=MOCK_SCENARIO):
            res = client.post(
                "/generate-scenario",
                json={"scenarioContext": "카페 주문", "goal": "커피 주문하기"},
            )
        levels = res.json()["data"]["levels"]
        assert len(levels) == 3

    def test_level_structure(self):
        with patch("api.app.generate_scenario_levels", return_value=MOCK_SCENARIO):
            res = client.post(
                "/generate-scenario",
                json={"scenarioContext": "카페 주문", "goal": "커피 주문하기"},
            )
        for level in res.json()["data"]["levels"]:
            assert "levelTitle" in level
            assert "levelDescription" in level
            assert "steps" in level
            assert len(level["steps"]) == 3

    def test_step_structure(self):
        with patch("api.app.generate_scenario_levels", return_value=MOCK_SCENARIO):
            res = client.post(
                "/generate-scenario",
                json={"scenarioContext": "카페 주문", "goal": "커피 주문하기"},
            )
        for level in res.json()["data"]["levels"]:
            for step in level["steps"]:
                assert "step" in step
                assert "assistantMessage" in step
                assert "userIntent" in step

    def test_required_fields(self):
        res = client.post("/generate-scenario", json={"scenarioContext": "카페"})
        assert res.status_code == 422

    @pytest.mark.skipif(not USE_REAL_OPENAI, reason="TEST_USE_REAL_OPENAI=1 로 활성화")
    def test_real_openai(self):
        res = client.post(
            "/generate-scenario",
            json={"scenarioContext": "카페 주문", "goal": "아메리카노 주문하기"},
        )
        assert res.status_code == 200
        assert res.json()["success"] is True
        assert len(res.json()["data"]["levels"]) == 3


# ── /practice/reference ───────────────────────────────────────────────────────

class TestReferencePracticeEndpoint:
    def _call(self, synthetic_wav, stt_text=MOCK_STT, ref_text="안녕하세요 반갑습니다"):
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value=stt_text), \
             patch("services.score_service._get_acoustic_text", return_value=None):
            return client.post(
                "/practice/reference",
                json={"s3Url": "https://mock.s3/audio.wav", "referenceText": ref_text},
            )

    def test_success_flag(self, synthetic_wav):
        res = self._call(synthetic_wav)
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_score_fields_present(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert "pronunciationScore" in data

    def test_removed_fields_absent(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        for field in ["meaningDeliveryScore", "wer", "cer", "similarityScore",
                      "diffAnalysis", "insertions", "mode", "referenceSource", "evaluationMode"]:
            assert field not in data, f"제거됐어야 할 필드: {field}"

    def test_score_type(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert isinstance(data["pronunciationScore"], (int, float))

    def test_score_range(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert 0.0 <= data["pronunciationScore"] <= 100.0

    def test_perfect_score_when_exact_match(self, synthetic_wav):
        """STT 결과가 referenceText와 완전히 일치하면 100점."""
        res = self._call(synthetic_wav, stt_text="안녕하세요", ref_text="안녕하세요")
        assert res.json()["pronunciationScore"] == 100.0

    def test_lower_score_on_mismatch(self, synthetic_wav):
        """STT 결과가 referenceText와 다르면 100점 미만."""
        res = self._call(synthetic_wav, stt_text="감사합니다", ref_text="안녕하세요")
        assert res.json()["pronunciationScore"] < 100.0

    def test_word_analysis_is_list(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert isinstance(data["wordAnalysis"], list)

    def test_word_analysis_item_keys(self, synthetic_wav):
        """wordAnalysis 항목은 refChar, hypChar, grade 만 포함."""
        data = self._call(synthetic_wav, stt_text="안녕", ref_text="안녕").json()
        for item in data["wordAnalysis"]:
            assert set(item.keys()) == {"refChar", "hypChar", "grade"}
            assert item["grade"] in ("good", "warn", "error")

    def test_voice_analysis_present(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        voice = data["voiceAnalysis"]
        assert "speechRate" in voice
        assert "silenceRatio" in voice

    def test_voice_analysis_grades(self, synthetic_wav):
        voice = self._call(synthetic_wav).json()["voiceAnalysis"]
        assert voice["speechRate"]["grade"] in ("good", "slow", "fast")
        assert "label" in voice["speechRate"]
        assert voice["silenceRatio"]["grade"] in ("good", "warn", "error")
        assert "label" in voice["silenceRatio"]

    def test_feedback_is_nonempty_string(self, synthetic_wav):
        feedback = self._call(synthetic_wav).json()["feedback"]
        assert isinstance(feedback, str) and len(feedback) > 0

    def test_required_fields_validation(self):
        res = client.post("/practice/reference", json={"s3Url": "https://mock.s3/audio.wav"})
        assert res.status_code == 422

    @pytest.mark.parametrize("ref,stt,should_be_perfect", [
        ("물 좀 주세요", "물 좀 주세요",   True),
        ("약 주세요",    "야 주세요",       False),
        ("안녕하세요",   "안녕",           False),
    ])
    def test_known_cases(self, synthetic_wav, ref, stt, should_be_perfect):
        res = self._call(synthetic_wav, stt_text=stt, ref_text=ref)
        score = res.json()["pronunciationScore"]
        if should_be_perfect:
            assert score == 100.0
        else:
            assert score < 100.0


# ── /practice/scenario ────────────────────────────────────────────────────────

class TestScenarioPracticeEndpoint:
    _PAYLOAD = {
        "s3Url": "https://mock.s3/audio.wav",
        "levelTitle": "병원 접수하기",
        "step": "용무 말하기",
        "assistantMessage": "어떻게 오셨나요?",
        "userIntent": "진료를 보러 왔다고 말해보세요.",
    }

    def _call(self, synthetic_wav, stt_text="진료 보러 왔어요", payload=None):
        data = payload or self._PAYLOAD
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value=stt_text), \
             patch("services.score_service.evaluate_with_llm", return_value=MOCK_SCENARIO_EVAL):
            return client.post("/practice/scenario", json=data)

    def test_success_flag(self, synthetic_wav):
        assert self._call(synthetic_wav).json()["success"] is True

    def test_echo_input_fields(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert data["levelTitle"] == "병원 접수하기"
        assert data["step"] == "용무 말하기"

    def test_removed_fields_absent(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        for field in ("wer", "cer", "similarityScore", "diffAnalysis", "insertions",
                      "mode", "referenceSource", "evaluationMode", "assistantMessage", "userIntent"):
            assert field not in data, f"제거됐어야 할 필드: {field}"

    def test_score_fields_present(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        for field in ["pronunciationScore", "meaningDeliveryScore"]:
            assert field in data

    def test_score_ranges(self, synthetic_wav):
        data = self._call(synthetic_wav).json()
        assert 0.0 <= data["pronunciationScore"] <= 100.0
        assert 0.0 <= data["meaningDeliveryScore"] <= 100.0

    def test_voice_analysis_present(self, synthetic_wav):
        voice = self._call(synthetic_wav).json()["voiceAnalysis"]
        assert "speechRate" in voice
        assert "silenceRatio" in voice

    def test_voice_analysis_grades(self, synthetic_wav):
        voice = self._call(synthetic_wav).json()["voiceAnalysis"]
        assert voice["speechRate"]["grade"] in ("good", "slow", "fast")
        assert "label" in voice["speechRate"]
        assert voice["silenceRatio"]["grade"] in ("good", "warn", "error")
        assert "label" in voice["silenceRatio"]

    def test_word_analysis_is_list(self, synthetic_wav):
        assert isinstance(self._call(synthetic_wav).json()["wordAnalysis"], list)

    def test_feedback_is_nonempty_string(self, synthetic_wav):
        feedback = self._call(synthetic_wav).json()["feedback"]
        assert isinstance(feedback, str) and len(feedback) > 0

    def test_required_fields_validation(self):
        res = client.post("/practice/scenario", json={"s3Url": "https://mock.s3/audio.wav"})
        assert res.status_code == 422

    def test_llm_meaning_score_used(self, synthetic_wav):
        """MOCK_SCENARIO_EVAL의 meaningDeliveryScore=85 가 그대로 반환되는지 확인."""
        data = self._call(synthetic_wav).json()
        assert data["meaningDeliveryScore"] == 85

    @pytest.mark.skipif(not USE_REAL_OPENAI, reason="TEST_USE_REAL_OPENAI=1 로 활성화")
    def test_real_openai(self, synthetic_wav):
        with patch("api.app.download_audio_from_s3", side_effect=make_s3_mock(synthetic_wav)), \
             patch("api.app.preprocess_audio_to_mono_16k_wav", side_effect=_mock_preprocess), \
             patch("api.app.transcribe_audio", return_value="진료 예약하러 왔어요"):
            res = client.post("/practice/scenario", json=self._PAYLOAD)
        assert res.status_code == 200
        assert res.json()["success"] is True


# ── /chat/free-talk ───────────────────────────────────────────────────────────

class TestFreeTalkEndpoint:
    def test_success_flag(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "오늘 날씨가 좋네요"})
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_mode_field(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "안녕"})
        assert res.json()["mode"] == "free_talk"

    def test_response_keys(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "안녕"})
        data = res.json()
        for key in ["success", "mode", "userMessage", "aiReply", "aiFeedback", "assistantMessageForHistory"]:
            assert key in data

    def test_echoes_user_message(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "테스트 메시지입니다"})
        assert res.json()["userMessage"] == "테스트 메시지입니다"

    def test_ai_reply_is_nonempty(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "안녕"})
        assert len(res.json()["aiReply"]) > 0

    def test_without_history(self):
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post("/chat/free-talk", json={"userMessage": "처음이에요"})
        assert res.status_code == 200

    def test_with_history(self):
        history = [
            {"role": "user",      "content": "안녕하세요"},
            {"role": "assistant", "content": "안녕하세요! 반갑습니다."},
        ]
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK):
            res = client.post(
                "/chat/free-talk",
                json={"userMessage": "오늘 기분이 좋아요", "chatHistory": history},
            )
        assert res.status_code == 200

    def test_history_passed_to_service(self):
        """chatHistory 가 서비스 함수에 올바르게 전달되는지 확인."""
        history = [{"role": "user", "content": "안녕"}]
        with patch("api.app.generate_free_talk_reply", return_value=MOCK_FREE_TALK) as mock_fn:
            client.post(
                "/chat/free-talk",
                json={"userMessage": "반가워요", "chatHistory": history},
            )
        _, kwargs = mock_fn.call_args
        assert kwargs.get("chat_history") == history

    def test_user_message_required(self):
        res = client.post("/chat/free-talk", json={})
        assert res.status_code == 422

    @pytest.mark.skipif(not USE_REAL_OPENAI, reason="TEST_USE_REAL_OPENAI=1 로 활성화")
    def test_real_openai(self):
        res = client.post("/chat/free-talk", json={"userMessage": "오늘 날씨가 참 좋네요"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["aiReply"]) > 0
        assert len(data["aiFeedback"]) > 0
