import librosa
import numpy as np
from typing import Dict, Any


def count_korean_syllables(text: str) -> int:
    """한글 음절 수 계산 (각 한글 문자 = 1음절)"""
    return sum(1 for ch in text if '가' <= ch <= '힣')


def analyze_speech_rate(y: np.ndarray, sr: int, stt_text: str) -> Dict[str, Any]:
    """
    조음속도 분석 (pause 제거 후 음절 수 ÷ 발화 시간)
    정상: 4.0–7.0 sps / slow: < 4.0 / fast: > 7.0
    점수: 4.0·7.0 sps → 70점, 5.2–5.9 sps → 100점, 9.0+ sps / 0 sps → 0점
    (Lee et al. 2017; Yoo et al. 2019; ASHA 2003)
    """
    syllable_count = count_korean_syllables(stt_text)

    if syllable_count == 0:
        return {"syllablesPerSecond": 0.0, "score": 0, "grade": "slow", "label": "측정 불가"}

    intervals = librosa.effects.split(y, top_db=35)
    voiced_duration = sum((end - start) / sr for start, end in intervals)

    if voiced_duration <= 0:
        return {"syllablesPerSecond": 0.0, "score": 0, "grade": "slow", "label": "측정 불가"}

    rate = round(syllable_count / voiced_duration, 2)

    if rate < 4.0:
        score = max(0, round(rate / 4.0 * 70))
        grade = "slow"
        label = "발화 속도가 느려요"
    elif rate <= 5.2:
        score = round(70 + (rate - 4.0) / 1.2 * 30)
        grade = "good"
        label = "적절한 속도예요"
    elif rate <= 5.9:
        score = 100
        grade = "good"
        label = "적절한 속도예요"
    elif rate <= 7.0:
        score = round(100 - (rate - 5.9) / 1.1 * 30)
        grade = "good"
        label = "적절한 속도예요"
    else:
        score = max(0, round(70 * (9.0 - rate) / 2.0))
        grade = "fast"
        label = "발화 속도가 빨라요"

    return {"syllablesPerSecond": rate, "score": score, "grade": grade, "label": label}


def analyze_silence_ratio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Pause 비율 분석 — 임상적 유의 pause(≥ 250ms) 점유 비율
    - good  : ≤ 25%
    - warn  : 25~45%
    - error : > 45%  (말 막힘 의심)
    (Angelopoulou et al., Brain Sciences 2024; Barnett et al., 2020)
    """
    PAUSE_MIN_SEC = 0.250  # 임상적 유의 pause 기준 (Angelopoulou et al., 2024)

    total_duration = len(y) / sr
    if total_duration <= 0:
        return {"pausePercent": 0.0, "grade": "good", "label": "자연스러운 흐름이에요"}

    intervals = librosa.effects.split(y, top_db=35)

    if len(intervals) == 0:
        pause_duration = total_duration
    else:
        pause_duration = 0.0

        leading = intervals[0][0] / sr
        if leading >= PAUSE_MIN_SEC:
            pause_duration += leading

        for i in range(len(intervals) - 1):
            gap = (intervals[i + 1][0] - intervals[i][1]) / sr
            if gap >= PAUSE_MIN_SEC:
                pause_duration += gap

        trailing = (len(y) - intervals[-1][1]) / sr
        if trailing >= PAUSE_MIN_SEC:
            pause_duration += trailing

    pause_ratio = round(max(0.0, min(1.0, pause_duration / total_duration)), 4)

    if pause_ratio <= 0.25:
        grade = "good"
        label = "자연스러운 흐름이에요"
    elif pause_ratio <= 0.45:
        grade = "warn"
        label = "쉬는 구간이 조금 많아요"
    else:
        grade = "error"
        label = "말 막힘이 의심돼요"

    return {"pausePercent": round(pause_ratio * 100, 1), "grade": grade, "label": label}


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
