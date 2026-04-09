from openai import OpenAI
from config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def text_to_speech(text: str, voice: str = "nova", speed: float = 1.0) -> bytes:
    """
    텍스트를 음성(mp3)으로 변환하여 bytes로 반환.

    Parameters:
        text  : 변환할 문자열
        voice : alloy | echo | fable | onyx | nova | shimmer (기본: nova)
        speed : 0.25 ~ 4.0 (기본: 1.0)

    Returns:
        mp3 오디오 bytes
    """
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        speed=speed
    )
    return response.content
