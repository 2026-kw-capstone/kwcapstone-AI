import librosa
import numpy as np
from typing import Dict, Any, Optional, Tuple


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



# ── 단모음 발음 정확도 분석 ────────────────────────────────────────────────────

# 한국어 단모음 기준 포먼트 (F1, F2) in Hz
# 출처: 한국어 음성학 연구 (성인 평균치 기준)
_KOREAN_VOWEL_FORMANTS: Dict[str, Tuple[float, float]] = {
    "아": (780, 1230),
    "이": (310, 2590),
    "우": (335,  840),
    "에": (500, 1870),
    "오": (490,  890),
    "애": (690, 1790),
    "외": (510, 1590),
    "위": (310, 2050),
    "으": (460, 1280),
    "의": (460, 1280),
}


def _extract_formants(y: np.ndarray, sr: int) -> Optional[Tuple[float, float]]:
    """
    LPC로 F1, F2 포먼트 추정.
    최소 50ms 이상 신호가 필요하며, 실패 시 None 반환.
    """
    if len(y) < int(sr * 0.05):
        return None
    try:
        # Pre-emphasis → 고주파 성분 강조
        y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
        y_win = y_pre * np.hamming(len(y_pre))

        order = int(2 + sr / 1000)  # 16kHz → order=18 (표준)
        coeffs = librosa.lpc(y_win, order=order)
        roots = np.roots(coeffs)

        # 양의 허수부 루트만 유지 (양의 주파수)
        roots = roots[np.imag(roots) >= 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = sorted([a * sr / (2 * np.pi) for a in angles if a > 0])
        freqs = [f for f in freqs if 50 < f < 5000]

        return (freqs[0], freqs[1]) if len(freqs) >= 2 else None
    except Exception:
        return None


def analyze_vowel_pronunciation(y: np.ndarray, sr: int, target_vowel: str) -> Dict[str, Any]:
    """
    단모음 발음 정확도 분석 (LPC 포먼트 기반, 0~100점)

    F1 허용 오차 300Hz, F2 허용 오차 600Hz 기준 선형 감점:
      score = max(0, 100 - |ΔF1|/3) × 0.5 + max(0, 100 - |ΔF2|/6) × 0.5
    """
    if target_vowel not in _KOREAN_VOWEL_FORMANTS:
        return {
            "score": 0, "grade": "error", "label": "지원하지 않는 모음이에요",
            "measuredF1": None, "measuredF2": None,
        }

    ref_f1, ref_f2 = _KOREAN_VOWEL_FORMANTS[target_vowel]

    intervals = librosa.effects.split(y, top_db=35)
    if len(intervals) == 0:
        return {
            "score": 0, "grade": "error", "label": "발성이 감지되지 않았어요",
            "measuredF1": None, "measuredF2": None,
        }

    # 가장 긴 유성 구간의 중간 60% 사용 (onset/offset 제거)
    start, end = max(intervals, key=lambda x: x[1] - x[0])
    n = end - start
    seg_s = start + int(n * 0.2)
    seg_e = start + int(n * 0.8)
    y_stable = y[seg_s:seg_e] if seg_e > seg_s else y[start:end]

    formants = _extract_formants(y_stable, sr)

    if formants is None:
        return {
            "score": 50, "grade": "warn", "label": "포먼트 측정이 불안정해요",
            "measuredF1": None, "measuredF2": None,
        }

    user_f1, user_f2 = formants
    f1_score = max(0.0, 100 - abs(user_f1 - ref_f1) / 3)   # 300Hz → 0
    f2_score = max(0.0, 100 - abs(user_f2 - ref_f2) / 6)   # 600Hz → 0
    score = round(0.5 * f1_score + 0.5 * f2_score)

    if score >= 75:
        grade, label = "good",  "모음 발음이 정확해요"
    elif score >= 50:
        grade, label = "warn",  "모음 발음이 조금 어긋났어요"
    else:
        grade, label = "error", "모음 발음을 다시 연습해보세요"

    return {
        "score": score, "grade": grade, "label": label,
        "measuredF1": round(user_f1), "measuredF2": round(user_f2),
    }


def analyze_vowel_voice(audio_path: str, target_vowel: str) -> Dict[str, Any]:
    """
    단모음 오디오에 대한 발음 정확도 분석.
    audio_path: 전처리 완료된 mono 16k wav 경로
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    return {
        "pronunciation": analyze_vowel_pronunciation(y, sr, target_vowel),
    }
