import os
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from config.settings import COLAB_API_TOKEN
from services.scenario_service import generate_scenario_levels
from services.audio_service import download_audio_from_s3, preprocess_audio_to_mono_16k_wav
from services.stt_service import transcribe_audio
from services.score_service import evaluate_reference_response, evaluate_scenario_response
from services.chat_service import generate_free_talk_reply
from services.tts_service import text_to_speech
from services.voice_analysis_service import analyze_voice, analyze_syllable_voice

app = FastAPI()

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class STTRequest(BaseModel):
    s3Url: str


class ScenarioRequest(BaseModel):
    scenarioContext: str
    goal: str


class ReferencePracticeRequest(BaseModel):
    s3Url: str
    referenceText: str


class FreeTalkRequest(BaseModel):
    userMessage: str
    chatHistory: Optional[list] = None


class ScenarioPracticeRequest(BaseModel):
    s3Url: str
    levelTitle: str
    step: str
    assistantMessage: str
    userIntent: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"   # alloy | echo | fable | onyx | nova | shimmer
    speed: float = 1.0    # 0.25 ~ 4.0


class SyllableVoiceRequest(BaseModel):
    s3Url: str


def validate_token(x_api_token: Optional[str]):
    if COLAB_API_TOKEN and x_api_token != COLAB_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stt")
def stt(req: STTRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    raw_path = os.path.join(INPUT_DIR, "stt_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "stt_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)
        stt_text = transcribe_audio(wav_path)

        return {
            "success": True,
            "sttText": stt_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
def tts(req: TTSRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)
    try:
        audio_bytes = text_to_speech(req.text, req.voice, req.speed)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-scenario")
def generate_scenario(req: ScenarioRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    try:
        result = generate_scenario_levels(req.scenarioContext, req.goal)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/practice/reference")
def practice_reference(req: ReferencePracticeRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    raw_path = os.path.join(INPUT_DIR, "reference_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "reference_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)
        stt_text = transcribe_audio(wav_path)

        eval_result = evaluate_reference_response(
            reference_text=req.referenceText,
            stt_text=stt_text,
            audio_path=wav_path
        )

        voice_result = analyze_voice(wav_path, stt_text)

        return {
            "success": True,
            **eval_result,
            "voiceAnalysis": voice_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/free-talk")
def free_talk(req: FreeTalkRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    try:
        result = generate_free_talk_reply(
            user_message=req.userMessage,
            chat_history=req.chatHistory
        )

        return {
            "success": True,
            "mode": "free_talk",
            "userMessage": req.userMessage,
            "aiReply": result["reply"],
            "aiFeedback": result["feedback"],
            "assistantMessageForHistory": result["assistant_message_for_history"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/practice/syllable-voice")
def syllable_voice(req: SyllableVoiceRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    raw_path = os.path.join(INPUT_DIR, "syllable_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "syllable_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)
        result = analyze_syllable_voice(wav_path)

        return {
            "success": True,
            "vocalizationDuration": result["vocalizationDuration"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/practice/scenario")
def practice_scenario(req: ScenarioPracticeRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    raw_path = os.path.join(INPUT_DIR, "scenario_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "scenario_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)
        stt_text = transcribe_audio(wav_path)

        step_content = f"AI 질문: {req.assistantMessage}\n사용자 연습 목표: {req.userIntent}"
        eval_result = evaluate_scenario_response(step_content, stt_text, audio_path=wav_path)

        voice_result = analyze_voice(wav_path, stt_text)

        return {
            "success": True,
            "levelTitle": req.levelTitle,
            "step": req.step,
            **eval_result,
            "voiceAnalysis": voice_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
