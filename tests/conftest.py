"""
공유 pytest 픽스처 및 테스트용 합성 오디오 생성.

합성 오디오 사양:
  - 포맷: mono, 16kHz, 16-bit PCM WAV
  - 라이브러리 의존성 없음 (stdlib: wave, struct, math 만 사용)

실제 음성 파일로 STT를 테스트하려면:
  tests/audio/speech.wav  에 파일을 직접 복사하면 됩니다.
"""

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
AUDIO_DIR = TESTS_DIR / "audio"


# ── 오디오 생성 헬퍼 ──────────────────────────────────────────────────────────

def _write_wav(
    path: str,
    duration: float = 2.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
    silence_start: float = 0.7,
    silence_end: float = 1.0,
) -> str:
    """
    stdlib 만 사용해 사인파 + 침묵 구간이 포함된 WAV 파일을 생성합니다.

    amplitude=0.3  →  약 -11 dBFS
    silence 구간   →  침묵 비율 계산용
    """
    n = int(sample_rate * duration)
    s_start = int(sample_rate * silence_start)
    s_end = int(sample_rate * silence_end)

    frames = bytearray()
    for i in range(n):
        if s_start <= i < s_end:
            val = 0
        else:
            val = int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", max(-32768, min(32767, val)))

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))

    return path


# ── 세션 픽스처 ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _ensure_audio_dir():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def synthetic_wav() -> str:
    """
    일반적인 합성 WAV.
    침묵 비율 적당. voice_analysis / API 테스트에서 범용으로 사용.
    """
    path = str(AUDIO_DIR / "synthetic.wav")
    _write_wav(path, duration=2.0, amplitude=0.3, silence_start=0.7, silence_end=1.0)
    return path


@pytest.fixture(scope="session")
def quiet_wav() -> str:
    """매우 작은 음량(~-55 dBFS) 합성 WAV."""
    path = str(AUDIO_DIR / "quiet.wav")
    _write_wav(path, duration=2.0, amplitude=0.003, silence_start=0.9, silence_end=1.0)
    return path


@pytest.fixture(scope="session")
def mostly_silent_wav() -> str:
    """침묵 구간이 전체의 ~80% → silenceRatio 'error' 유발."""
    path = str(AUDIO_DIR / "mostly_silent.wav")
    _write_wav(
        path,
        duration=5.0,
        amplitude=0.3,
        silence_start=0.5,   # 0.5s 발화
        silence_end=4.5,     # 4.0s 침묵
    )
    return path


@pytest.fixture(scope="session")
def real_speech_wav():
    """
    사용자가 직접 넣은 실제 음성 파일.
    tests/audio/speech.wav 가 존재하면 경로 반환, 없으면 None.

    STT 실제 변환 테스트(TEST_USE_REAL_WHISPER=1)에서 사용됩니다.
    """
    path = AUDIO_DIR / "speech.wav"
    return str(path) if path.exists() else None


# ── S3 다운로드 mock 헬퍼 ─────────────────────────────────────────────────────

def make_s3_mock(local_wav_path: str):
    """
    download_audio_from_s3 를 로컬 파일 복사로 대체하는 mock factory.

    사용 예:
        with patch("api.app.download_audio_from_s3",
                   side_effect=make_s3_mock(synthetic_wav)):
            ...
    """
    def _mock(s3_url: str, save_path: str) -> str:
        shutil.copy2(local_wav_path, save_path)
        return save_path
    return _mock
