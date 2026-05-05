import librosa
import numpy as np
from typing import Dict, Any


def count_korean_syllables(text: str) -> int:
    """한글 음절 수 계산 (각 한글 문자 = 1음절)"""
    return sum(1 for ch in text if '가' <= ch <= '힣')


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
        return {"syllablesPerSecond": 0.0, "grade": "error", "label": "측정 불가"}

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

    return {"syllablesPerSecond": rate, "grade": grade, "label": label}


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

    return {"silencePercent": round(silence_ratio * 100, 1), "grade": grade, "label": label}


def analyze_voice(audio_path: str, stt_text: str = "") -> Dict[str, Any]:
    """
    음성 파일을 분석하여 음량 / 발화속도 / 침묵비율 반환
    audio_path: 전처리 완료된 mono 16k wav 경로
    stt_text  : 발화속도 계산용 STT 결과 텍스트
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    return {
        "speechRate": analyze_speech_rate(y, sr, stt_text),
        "silenceRatio": analyze_silence_ratio(y, sr)
    }


def analyze_vocalization_duration(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    한 음절 발성 시간 분석 (무음 제거 후 실제 발성 구간 합산 → 0~100점)
    0~0.3초: 0~50점 선형, 0.3~2.0초: 50~100점 선형, 2.0초 이상: 100점
    """
    intervals = librosa.effects.split(y, top_db=35)
    voiced_duration = sum((end - start) / sr for start, end in intervals)

    if voiced_duration <= 0:
        score = 0
    elif voiced_duration < 0.3:
        score = round(voiced_duration / 0.3 * 50)
    elif voiced_duration < 2.0:
        score = round(50 + (voiced_duration - 0.3) / 1.7 * 50)
    else:
        score = 100

    if score >= 75:
        grade = "good"
        label = "발성 시간이 충분해요"
    elif score >= 40:
        grade = "warn"
        label = "조금 더 길게 발성해 보세요"
    else:
        grade = "error"
        label = "발성 시간이 너무 짧아요"

    return {"durationSeconds": round(voiced_duration, 3), "score": score, "grade": grade, "label": label}


def analyze_syllable_voice(audio_path: str) -> Dict[str, Any]:
    """
    한 음절 오디오에 대한 음량 + 발성 시간 분석
    audio_path: 전처리 완료된 mono 16k wav 경로
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    return {
        "vocalizationDuration": analyze_vocalization_duration(y, sr)
    }
