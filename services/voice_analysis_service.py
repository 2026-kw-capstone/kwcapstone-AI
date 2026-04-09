import librosa
import numpy as np
from typing import Dict, Any


def count_korean_syllables(text: str) -> int:
    """한글 음절 수 계산 (각 한글 문자 = 1음절)"""
    return sum(1 for ch in text if '가' <= ch <= '힣')


def analyze_loudness(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    음량 분석 (RMS → dBFS)
    - good  : > -25 dBFS  (적절한 크기)
    - warn  : -35 ~ -25   (조금 작은 편)
    - error : < -35 dBFS  (너무 작음)
    """
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))

    avg_db = float(20 * np.log10(mean_rms)) if mean_rms > 0 else -80.0

    if avg_db >= -25:
        grade = "good"
        label = "적절한 크기예요"
    elif avg_db >= -35:
        grade = "warn"
        label = "목소리가 조금 작아요"
    else:
        grade = "error"
        label = "목소리가 너무 작아요"

    return {
        "avgDb": round(avg_db, 2),
        "grade": grade,
        "label": label
    }


def analyze_speech_rate(y: np.ndarray, sr: int, stt_text: str) -> Dict[str, Any]:
    """
    발화 속도 분석 (음절/초)
    한국어 정상 발화 속도: 4~6음절/초
    - good  : 3.0 ~ 6.0 음절/초
    - warn  : 2.0 ~ 3.0 또는 6.0 ~ 7.5
    - error : < 2.0 또는 > 7.5
    """
    duration = librosa.get_duration(y=y, sr=sr)
    syllable_count = count_korean_syllables(stt_text)

    if duration <= 0 or syllable_count == 0:
        return {
            "syllablesPerSecond": 0.0,
            "syllableCount": syllable_count,
            "durationSeconds": round(duration, 2),
            "grade": "error",
            "label": "측정 불가"
        }

    rate = round(syllable_count / duration, 2)

    if 3.0 <= rate <= 6.0:
        grade = "good"
        label = "적절한 속도예요"
    elif 2.0 <= rate < 3.0:
        grade = "warn"
        label = "조금 느린 편이에요"
    elif 6.0 < rate <= 7.5:
        grade = "warn"
        label = "조금 빠른 편이에요"
    elif rate < 2.0:
        grade = "error"
        label = "너무 느린 편이에요"
    else:
        grade = "error"
        label = "너무 빠른 편이에요"

    return {
        "syllablesPerSecond": rate,
        "syllableCount": syllable_count,
        "durationSeconds": round(duration, 2),
        "grade": grade,
        "label": label
    }


def analyze_silence_ratio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    침묵 비율 분석 (무음 구간 / 전체 시간)
    - good  : <= 25%  (자연스러운 흐름)
    - warn  : 25 ~ 45% (쉬는 구간이 많음)
    - error : > 45%   (말 막힘 의심)

    top_db=35: 최대 에너지 대비 35dB 이상 낮은 구간을 무음으로 판단
    """
    total_duration = len(y) / sr

    intervals = librosa.effects.split(y, top_db=35)
    speaking_duration = sum((end - start) / sr for start, end in intervals)
    silence_duration = total_duration - speaking_duration

    silence_ratio = silence_duration / total_duration if total_duration > 0 else 0.0
    silence_ratio = round(max(0.0, min(1.0, silence_ratio)), 4)

    if silence_ratio <= 0.25:
        grade = "good"
        label = "자연스러운 흐름이에요"
    elif silence_ratio <= 0.45:
        grade = "warn"
        label = "쉬는 구간이 조금 많아요"
    else:
        grade = "error"
        label = "말 막힘이 의심돼요"

    return {
        "silenceRatio": silence_ratio,
        "silencePercent": round(silence_ratio * 100, 1),
        "grade": grade,
        "label": label
    }


def analyze_voice(audio_path: str, stt_text: str = "") -> Dict[str, Any]:
    """
    음성 파일을 분석하여 음량 / 발화속도 / 침묵비율 반환
    audio_path: 전처리 완료된 mono 16k wav 경로
    stt_text  : 발화속도 계산용 STT 결과 텍스트
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    return {
        "loudness": analyze_loudness(y, sr),
        "speechRate": analyze_speech_rate(y, sr, stt_text),
        "silenceRatio": analyze_silence_ratio(y, sr)
    }
