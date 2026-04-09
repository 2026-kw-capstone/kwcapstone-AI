import whisper

model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model("small") #pycharm에서 실행 시 -> 코랩에선 large로 수정할 예정
    return model

def transcribe_audio(audio_path: str) -> str:
    stt_model = get_model()
    result = stt_model.transcribe(audio_path, language="ko")
    return result["text"].strip()
