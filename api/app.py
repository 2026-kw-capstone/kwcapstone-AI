import os
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from config.settings import COLAB_API_TOKEN
from services.scenario_service import generate_scenario_levels, regenerate_scenario_from_step
from services.scenario_agent_service import start_session_turn, advance_session_turn
from services import session_store
from services.audio_service import download_audio_from_s3, preprocess_audio_to_mono_16k_wav
from services.stt_service import transcribe_audio
from services.score_service import evaluate_reference_response, evaluate_scenario_response, generate_vowel_feedback
from services.chat_service import generate_free_talk_reply
from services.tts_service import text_to_speech
from services.voice_analysis_service import analyze_voice, analyze_vowel_voice

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


class RegenerateStepRequest(BaseModel):
    scenarioContext: str
    goal: str
    levels: list
    targetLevelIndex: int
    targetStepIndex: int


class ScenarioPracticeRequest(BaseModel):
    s3Url: str
    levelTitle: str
    step: str
    assistantMessage: str
    userIntent: str


class ScenarioSessionStartRequest(BaseModel):
    scenarioContext: str
    goal: str
    maxSteps: int = 9


class ScenarioSessionRespondRequest(BaseModel):
    sessionId: str
    s3Url: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"   # alloy | echo | fable | onyx | nova | shimmer
    speed: float = 1.0    # 0.25 ~ 4.0


class VowelPracticeRequest(BaseModel):
    s3Url: str
    targetVowel: str  # "아" | "에" | "이" | "오" | "우"


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


@app.post("/regenerate-scenario-step")
def regenerate_step(req: RegenerateStepRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    try:
        result = regenerate_scenario_from_step(
            scenario_context=req.scenarioContext,
            goal=req.goal,
            levels=req.levels,
            target_level_index=req.targetLevelIndex,
            target_step_index=req.targetStepIndex,
        )
        return {
            "success": True,
            "data": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

        voice_result = analyze_voice(wav_path, stt_text)

        eval_result = evaluate_reference_response(
            reference_text=req.referenceText,
            stt_text=stt_text,
            audio_path=wav_path,
            voice_result=voice_result,
        )

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


@app.post("/practice/vowel")
def vowel_practice(req: VowelPracticeRequest, x_api_token: Optional[str] = Header(default=None)):
    validate_token(x_api_token)

    raw_path = os.path.join(INPUT_DIR, "vowel_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "vowel_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)

        result = analyze_vowel_voice(wav_path, req.targetVowel)
        pronunciation = result["pronunciation"]

        llm_result = generate_vowel_feedback(
            target_vowel=req.targetVowel,
            pronunciation=pronunciation,
        )

        return {
            "success": True,
            "targetVowel": req.targetVowel,
            "pronunciationScore": pronunciation["score"],
            "pronunciationGrade": pronunciation["grade"],
            "pronunciationLabel": pronunciation["label"],
            "feedback": llm_result.get("feedback", ""),
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

        voice_result = analyze_voice(wav_path, stt_text)

        step_content = f"AI 질문: {req.assistantMessage}\n사용자 연습 목표: {req.userIntent}"
        eval_result = evaluate_scenario_response(
            step_content, stt_text,
            audio_path=wav_path,
            voice_result=voice_result,
        )

        return {
            "success": True,
            "levelTitle": req.levelTitle,
            "step": req.step,
            **eval_result,
            "voiceAnalysis": voice_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent 기반 적응형 시나리오 세션 ─────────────────────────────────────────
@app.post("/scenario/session/start")
def scenario_session_start(
    req: ScenarioSessionStartRequest,
    x_api_token: Optional[str] = Header(default=None),
):
    """새 훈련 세션을 만들고 agent가 첫 step을 동적으로 생성한다."""
    validate_token(x_api_token)

    try:
        session = session_store.create_session(
            scenario_context=req.scenarioContext,
            goal=req.goal,
            max_steps=req.maxSteps,
        )
        turn = start_session_turn(session)
        session_store.save_session(session)

        return {
            "success": True,
            "sessionId": session["sessionId"],
            "status": session["status"],
            "currentLevel": session["currentLevel"],
            "stepInLevel": session["currentStepInLevel"],
            "stepsCompleted": session["stepsCompleted"],
            "maxSteps": session["maxSteps"],
            "step": turn.get("nextStep"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario/session/respond")
def scenario_session_respond(
    req: ScenarioSessionRespondRequest,
    x_api_token: Optional[str] = Header(default=None),
):
    """사용자 음성 답변을 받아 채점하고, agent가 다음 step을 결정한다."""
    validate_token(x_api_token)

    session = session_store.get_session(req.sessionId)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없음 (만료되었거나 잘못된 sessionId).")
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="이미 종료된 세션임.")
    if not session.get("currentStep"):
        raise HTTPException(status_code=400, detail="현재 답해야 할 step이 없음.")

    raw_path = os.path.join(INPUT_DIR, "scenario_session_input_audio")
    wav_path = os.path.join(OUTPUT_DIR, "scenario_session_input.wav")

    try:
        download_audio_from_s3(req.s3Url, raw_path)
        preprocess_audio_to_mono_16k_wav(raw_path, wav_path)
        stt_text = transcribe_audio(wav_path)

        voice_result = analyze_voice(wav_path, stt_text)

        turn = advance_session_turn(
            session=session,
            stt_text=stt_text,
            audio_path=wav_path,
            voice_result=voice_result,
        )
        session_store.save_session(session)

        evaluation = turn.get("lastEvaluation", {})
        return {
            "success": True,
            "sessionId": session["sessionId"],
            "status": session["status"],
            "sttText": stt_text,
            "evaluation": {
                "pronunciationScore": evaluation.get("pronunciationScore"),
                "meaningDeliveryScore": evaluation.get("meaningDeliveryScore"),
                "pronunciationFeedback": evaluation.get("pronunciationFeedback"),
                "meaningDeliveryFeedback": evaluation.get("meaningDeliveryFeedback"),
                "wordAnalysis": evaluation.get("wordAnalysis"),
            },
            "voiceAnalysis": voice_result,
            "difficulty": turn.get("difficulty"),
            "difficultyReason": turn.get("difficultyReason"),
            "stepsCompleted": session["stepsCompleted"],
            "maxSteps": session["maxSteps"],
            "currentLevel": session["currentLevel"],
            "stepInLevel": session["currentStepInLevel"],
            "nextStep": turn.get("nextStep"),
            "closingMessage": turn.get("closingMessage"),
            "finished": turn.get("finished", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scenario/session/{session_id}")
def scenario_session_get(
    session_id: str,
    x_api_token: Optional[str] = Header(default=None),
):
    """세션 상태와 진행 히스토리를 조회한다."""
    validate_token(x_api_token)

    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없음.")
    return {"success": True, **session}
