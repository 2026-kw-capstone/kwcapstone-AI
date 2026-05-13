import wave
import whisper

model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("small") #pycharm에서 실행 시 -> 코랩에선 large로 수정할 예정
    return model


def _get_wav_duration(audio_path: str) -> float:
    try:
        with wave.open(audio_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def transcribe_audio(audio_path: str) -> str:
    stt_model = get_model()
    result = stt_model.transcribe(
        audio_path,
        language="ko",
        temperature=0,                    # 결정론적 출력으로 할루시네이션 억제
        condition_on_previous_text=False, # 이전 구간 텍스트 조건화 비활성화 (핵심)
        no_speech_threshold=0.5,          # 무음 판정 기준 (기본 0.6보다 엄격하게)
        compression_ratio_threshold=2.4,  # 압축률 초과 시 할루시네이션으로 간주
    )

    text = result["text"].strip()

    # 오디오 길이 대비 음절 수 과다 → 할루시네이션 판정
    duration = _get_wav_duration(audio_path)
    if duration > 0:
        # 한국어 최대 발화 속도 ~8음절/초 기준
        max_syllables = int(duration * 8)
        actual_syllables = sum(1 for c in text if "가" <= c <= "힣")
        # 1초 미만 짧은 발화에서 음절 수가 기준의 1.5배 초과 시 할루시네이션
        if duration < 1.0 and actual_syllables > max_syllables * 1.5:
            return ""

    return text
