import requests
from pydub import AudioSegment


def download_audio_from_s3(s3_url: str, save_path: str):
    response = requests.get(s3_url, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


def preprocess_audio_to_mono_16k_wav(input_path: str, output_path: str):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path
