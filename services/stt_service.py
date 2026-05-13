import librosa
import whisper

model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("medium") #pycharm에서 실행 시 -> 코랩에선 large로 수정할 예정
    return model


def _get_voiced_duration(audio_path: str) -> float:
    """무음을 제외한 실제 발화 구간 길이 반환 (librosa VAD 기반)."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        intervals = librosa.effects.split(y, top_db=35)
        return sum((end - start) / sr for start, end in intervals)
    except Exception:
        return 0.0


def _has_repeated_suffix(text: str, min_len: int = 2) -> bool:
    """'세요세요', '습니다습니다' 같은 접미어 반복 할루시네이션 감지."""
    n = len(text)
    for length in range(min_len, n // 2 + 1):
        if text[n - 2 * length : n - length] == text[n - length:]:
            return True
    return False


def _count_korean_chars(text: str) -> int:
    """완성 음절(가-힣) + 자모(ㄱ-ㅣ) 모두 카운트."""
    return sum(
        1 for c in text
        if "가" <= c <= "힣" or "ㄱ" <= c <= "ㅣ"
    )


def transcribe_audio(audio_path: str) -> str:
    stt_model = get_model()
    result = stt_model.transcribe(
        audio_path,
        language="ko",
        temperature=0,                    # 결정론적 출력으로 할루시네이션 억제
        condition_on_previous_text=False, # 이전 구간 텍스트 조건화 비활성화 (핵심)
        no_speech_threshold=0.5,          # 무음 판정 기준 (기본 0.6보다 엄격하게)
        compression_ratio_threshold=2.4,  # 압축률 초과 시 할루시네이션으로 간주
        logprob_threshold=-0.5,           # 평균 log prob 낮으면 할루시네이션으로 간주 (기본 -1.0)
    )

    text = result["text"].strip()

    # 접미어 반복 패턴 감지 ("세요세요", "습니다습니다" 등)
    if _has_repeated_suffix(text):
        # 반복된 접미어를 제거하고 앞부분만 반환
        n = len(text)
        for length in range(2, n // 2 + 1):
            if text[n - 2 * length : n - length] == text[n - length:]:
                text = text[: n - length].strip()
                break

    # 실제 발화 시간(무음 제외) 대비 음절 수 과다 → 할루시네이션 판정
    voiced_duration = _get_voiced_duration(audio_path)
    if voiced_duration > 0:
        # 한국어 최대 발화 속도 ~8음절/초 기준
        max_syllables = int(voiced_duration * 8)
        actual_syllables = _count_korean_chars(text)
        # 발화 시간 기준 가능한 음절의 1.2배 초과 시 할루시네이션으로 판정
        threshold = 1.2 if voiced_duration < 1.0 else 1.5
        if actual_syllables > max_syllables * threshold:
            return ""

    return text
