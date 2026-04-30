"""
wav2vec2 기반 음향 음소 인식 서비스.

Whisper(seq2seq + 언어 모델)는 문맥 보정이 강해 실제 오발음을 정답으로 처리하는 경우가 있습니다.
wav2vec2(CTC 기반)는 언어 모델 보정 없이 오디오 파형에서 직접 음소를 추출하므로
아래 오류 유형을 더 정확히 감지합니다.

  ① 모음 왜곡  — "아"→"어": Whisper는 문맥상 "아"로 보정하지만 wav2vec2는 "어"로 출력
  ② 받침 약화  — "약"→"야": Whisper는 받침을 채워넣지만 wav2vec2는 누락 그대로 출력
  ③ 기식음 혼동 — ㅂ/ㅍ: wav2vec2는 acoustic feature(기식 에너지)로 구분
  ④ 음운 규칙  — G2P와 함께 사용 시 표기 ≠ 발음도 정확히 처리

사용 모델: kresnik/wav2vec2-large-xlsr-korean (HuggingFace)
미설치·GPU 없음·오류 시 None 반환 → 호출자에서 STT 결과로 자동 fallback.
"""

from typing import Optional

_MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
_processor = None
_model = None


def _get_model():
    global _processor, _model
    if _model is None:
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            _processor = Wav2Vec2Processor.from_pretrained(_MODEL_ID)
            _model = Wav2Vec2ForCTC.from_pretrained(_MODEL_ID)
            _model.eval()
        except Exception:
            pass
    return _processor, _model


def acoustic_recognize(audio_path: str) -> Optional[str]:
    """
    오디오 파일에서 CTC 기반으로 한국어 텍스트 인식.

    Whisper와의 차이:
    - 언어 모델 보정 없음 → 실제 발화된 음소를 그대로 반영
    - 받침 약화, 모음 왜곡, 기식음 혼동 등 미세 오류 감지 가능

    Args:
        audio_path: 전처리된 mono 16kHz WAV 파일 경로

    Returns:
        인식된 텍스트 (공백 제거 후). 모델 없거나 오류 시 None.
    """
    processor, model = _get_model()
    if processor is None or model is None:
        return None

    try:
        import torch
        import librosa

        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0]
        result = transcription.strip()
        return result if result else None

    except Exception:
        return None


def is_model_available() -> bool:
    """wav2vec2 모델 사용 가능 여부 확인 (테스트·디버깅용)."""
    processor, model = _get_model()
    return processor is not None and model is not None
