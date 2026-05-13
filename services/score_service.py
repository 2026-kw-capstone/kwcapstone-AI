import re
from difflib import SequenceMatcher
from jiwer import wer, cer
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
from typing import List, Dict, Any, Optional

client = OpenAI(api_key=OPENAI_API_KEY)

# 발음에 영향 없는 구두점·특수문자 제거용 패턴
# 한글 음절(가-힣), 한글 자모(ㄱ-ㅣ), 숫자, 영문자만 유지
_PUNCT_RE = re.compile(r'[^가-힣ㄱ-ㆎ0-9a-zA-Z]')


def split_korean_chars(text: str) -> List[str]:
    # 마침표·쉼표 등 구두점을 제거한 뒤 문자 단위로 분리
    cleaned = _PUNCT_RE.sub('', text)
    return [ch for ch in cleaned if ch.strip()]


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


CHOSUNG = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
JUNGSUNG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
JONGSUNG = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]


def decompose_hangul(char: str):
    if len(char) != 1 or not ('가' <= char <= '힣'):
        return None

    base = ord(char) - ord('가')
    cho = base // 588
    jung = (base % 588) // 28
    jong = base % 28

    return {
        "initial": CHOSUNG[cho],
        "medial": JUNGSUNG[jung],
        "final": JONGSUNG[jong]
    }


# ── G2P (Grapheme-to-Phoneme) ──────────────────────────────────────────────

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        try:
            from g2pk import G2p
            _g2p = G2p()
        except Exception:
            pass
    return _g2p


def apply_g2p(text: str) -> str:
    """한국어 텍스트에 음운 규칙 적용 (G2P)
    - 닭이 → 달기 (연음화)
    - 학교 → 학꾜 (경음화)
    - 국민 → 궁민 (비음화)
    g2pk 미설치·오류 시 원본 반환.
    """
    g2p = _get_g2p()
    if g2p is None or not text.strip():
        return text
    try:
        return g2p(text)
    except Exception:
        return text


def text_to_phoneme_sequence(text: str) -> List[Dict[str, Any]]:
    """텍스트(G2P 적용 후) → 음소 시퀀스 flat list
    각 항목: {phoneme, syl_idx, position('initial'|'medial'|'final')}
    종성 없는 음절은 초성+중성 2개만 포함.
    """
    seq = []
    for syl_idx, char in enumerate(split_korean_chars(text)):
        parts = decompose_hangul(char)
        if parts is None:
            seq.append({"phoneme": char, "syl_idx": syl_idx, "position": "initial"})
            continue
        seq.append({"phoneme": parts["initial"], "syl_idx": syl_idx, "position": "initial"})
        seq.append({"phoneme": parts["medial"],  "syl_idx": syl_idx, "position": "medial"})
        if parts["final"]:
            seq.append({"phoneme": parts["final"], "syl_idx": syl_idx, "position": "final"})
    return seq


def build_phoneme_based_analysis(ref_text: str, hyp_text: str):
    """G2P 음운 변환 + 음소 시퀀스 정렬 기반 발음 분석

    처리 흐름:
    1. 양측 텍스트에 G2P 적용 → 실제 발음 형태 획득
    2. 음소 시퀀스로 분해 (초/중/종성 flat list)
    3. SequenceMatcher로 음소 단위 정렬
    4. ref 음절별 오류 음소 위치(position) 집계
    5. 음절별 score/grade 산출 → word_analysis 반환 (원본 ref 음절 기준)

    반환: (word_analysis, inserts, insert_feedbacks)
    """
    ref_chars = split_korean_chars(ref_text)
    hyp_chars = split_korean_chars(hyp_text)

    ref_g2p = apply_g2p(ref_text)
    hyp_g2p = apply_g2p(hyp_text)

    ref_phonemes = text_to_phoneme_sequence(ref_g2p)
    hyp_phonemes = text_to_phoneme_sequence(hyp_g2p)

    ref_seq = [p["phoneme"] for p in ref_phonemes]
    hyp_seq = [p["phoneme"] for p in hyp_phonemes]

    n_ref_syls = len(split_korean_chars(ref_g2p))
    syl_err_positions: Dict[int, set] = {i: set() for i in range(n_ref_syls)}

    inserts: List[str] = []
    insert_feedbacks: List[str] = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref_seq, hyp_seq).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            for ri in range(i1, i2):
                ph = ref_phonemes[ri]
                syl_err_positions[ph["syl_idx"]].add(ph["position"])
        if tag == "insert":
            for ji in range(j1, j2):
                inserts.append(hyp_phonemes[ji]["phoneme"])
                insert_feedbacks.append("불필요한 발음이 추가되었어요")

    word_analysis = []
    for syl_idx, ref_char in enumerate(ref_chars):
        hyp_char = hyp_chars[syl_idx] if syl_idx < len(hyp_chars) else ""
        err_positions = syl_err_positions.get(syl_idx, set())

        if not hyp_char:
            score, grade, error_type = 0, "error", "delete"
            phoneme_diff = {"initial": "missing", "medial": "missing", "final": "missing"}
        else:
            diff_count = len(err_positions)
            score = {0: 100, 1: 70, 2: 40}.get(diff_count, 10)
            grade = "good" if score >= 100 else ("warn" if score >= 40 else "error")
            error_type = "+".join(sorted(err_positions)) if err_positions else "equal"
            phoneme_diff = {
                pos: ("different" if pos in err_positions else "same")
                for pos in ("initial", "medial", "final")
            }

        word_analysis.append({
            "refIndex":    syl_idx,
            "refChar":     ref_char,
            "hypChar":     hyp_char,
            "score":       score,
            "grade":       grade,
            "errorType":   error_type,
            "refParts":    decompose_hangul(ref_char),
            "hypParts":    decompose_hangul(hyp_char) if hyp_char else None,
            "phonemeDiff": phoneme_diff,
        })

    return word_analysis, inserts, insert_feedbacks


def analyze_syllable_difference(ref_char: str, hyp_char: str):
    ref_parts = decompose_hangul(ref_char)
    hyp_parts = decompose_hangul(hyp_char)

    if not ref_parts or not hyp_parts:
        return {
            "type": "unknown",
            "refParts": ref_parts,
            "hypParts": hyp_parts,
            "phonemeDiff": {
                "initial": "unknown",
                "medial": "unknown",
                "final": "unknown"
            }
        }

    diff_parts = []

    initial_state = "same" if ref_parts["initial"] == hyp_parts["initial"] else "different"
    medial_state = "same" if ref_parts["medial"] == hyp_parts["medial"] else "different"
    final_state = "same" if ref_parts["final"] == hyp_parts["final"] else "different"

    if initial_state == "different":
        diff_parts.append("initial")
    if medial_state == "different":
        diff_parts.append("medial")
    if final_state == "different":
        diff_parts.append("final")

    return {
        "type": "+".join(diff_parts) if diff_parts else "equal",
        "refParts": ref_parts,
        "hypParts": hyp_parts,
        "phonemeDiff": {
            "initial": initial_state,
            "medial": medial_state,
            "final": final_state
        }
    }



def calculate_reference_scores(reference_text: str, stt_text: str):
    reference_text = reference_text.strip()
    stt_text = stt_text.strip()

    if not reference_text:
        return {
            "pronunciationScore": 0.0,
            "similarityScore": 0.0,
            "wer": 1.0,
            "cer": 1.0
        }

    wer_score = wer(reference_text, stt_text)
    cer_score = cer(reference_text, stt_text)
    similarity = SequenceMatcher(None, reference_text, stt_text).ratio()

    pronunciation_score = round(max(0, 100 * (1 - wer_score)), 2)
    similarity_score = round(similarity * 100, 2)

    return {
        "pronunciationScore": pronunciation_score,
        "similarityScore": similarity_score,
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4)
    }


def build_alignment(reference_text: str, stt_text: str) -> Dict[str, Any]:
    reference_chars = split_korean_chars(reference_text)
    stt_chars = split_korean_chars(stt_text)

    matcher = SequenceMatcher(None, reference_chars, stt_chars)
    opcodes = matcher.get_opcodes()

    aligned = []
    mismatch_indexes = []
    ref_index = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                aligned.append({
                    "refIndex": ref_index,
                    "ref": reference_chars[i1 + k],
                    "hyp": stt_chars[j1 + k],
                    "type": "equal"
                })
                ref_index += 1

        elif tag == "replace":
            ref_chunk = reference_chars[i1:i2]
            hyp_chunk = stt_chars[j1:j2]
            max_len = max(len(ref_chunk), len(hyp_chunk))

            for k in range(max_len):
                ref_char = ref_chunk[k] if k < len(ref_chunk) else ""
                hyp_char = hyp_chunk[k] if k < len(hyp_chunk) else ""

                if ref_char and hyp_char:
                    aligned.append({
                        "refIndex": ref_index,
                        "ref": ref_char,
                        "hyp": hyp_char,
                        "type": "substitute"
                    })
                    mismatch_indexes.append(ref_index)
                    ref_index += 1

                elif ref_char and not hyp_char:
                    aligned.append({
                        "refIndex": ref_index,
                        "ref": ref_char,
                        "hyp": "",
                        "type": "delete"
                    })
                    mismatch_indexes.append(ref_index)
                    ref_index += 1

                elif not ref_char and hyp_char:
                    aligned.append({
                        "refIndex": max(ref_index - 1, 0),
                        "ref": "",
                        "hyp": hyp_char,
                        "type": "insert"
                    })

        elif tag == "delete":
            for k in range(i1, i2):
                aligned.append({
                    "refIndex": ref_index,
                    "ref": reference_chars[k],
                    "hyp": "",
                    "type": "delete"
                })
                mismatch_indexes.append(ref_index)
                ref_index += 1

        elif tag == "insert":
            for k in range(j1, j2):
                aligned.append({
                    "refIndex": max(ref_index - 1, 0),
                    "ref": "",
                    "hyp": stt_chars[k],
                    "type": "insert"
                })

    return {
        "referenceChars": reference_chars,
        "sttChars": stt_chars,
        "aligned": aligned,
        "mismatchIndexes": mismatch_indexes
    }


def build_rule_based_analysis(reference_text: str, alignment_result: Dict[str, Any]):
    aligned_pairs = alignment_result.get("aligned", [])

    word_analysis = []
    inserts = []
    insert_feedbacks = []

    for pair in aligned_pairs:
        ref_char = pair.get("ref", "")
        hyp_char = pair.get("hyp", "")
        pair_type = pair.get("type", "")

        if pair_type == "insert":
            inserts.append(hyp_char)
            insert_feedbacks.append(f"'{hyp_char}' 발화가 추가로 들어갔을 수 있음")
            continue

        item = {
            "refIndex": pair.get("refIndex"),
            "refChar": ref_char,
            "hypChar": hyp_char,
            "score": 0,
            "grade": "error",
            "errorType": None,
            "refParts": decompose_hangul(ref_char) if ref_char else None,
            "hypParts": decompose_hangul(hyp_char) if hyp_char else None,
            "phonemeDiff": {
                "initial": "same" if ref_char == hyp_char and ref_char else "unknown",
                "medial": "same" if ref_char == hyp_char and ref_char else "unknown",
                "final": "same" if ref_char == hyp_char and ref_char else "unknown"
            }
        }

        if pair_type == "equal":
            item["score"] = 100
            item["errorType"] = "equal"
            if item["refParts"] and item["hypParts"]:
                item["phonemeDiff"] = {"initial": "same", "medial": "same", "final": "same"}

        elif pair_type == "substitute":
            diff = analyze_syllable_difference(ref_char, hyp_char)
            diff_count = sum(1 for v in diff["phonemeDiff"].values() if v == "different")
            item["score"] = {0: 100, 1: 70, 2: 40, 3: 10}.get(diff_count, 10)
            item["errorType"] = diff["type"] if diff["type"] != "equal" else "equal"
            item["refParts"] = diff.get("refParts")
            item["hypParts"] = diff.get("hypParts")
            item["phonemeDiff"] = diff.get("phonemeDiff", {
                "initial": "unknown", "medial": "unknown", "final": "unknown"
            })
            if diff["type"] == "equal":
                item["score"] = 100

        elif pair_type == "delete":
            item["score"] = 0
            item["errorType"] = "delete"
            item["hypParts"] = None
            item["phonemeDiff"] = {"initial": "missing", "medial": "missing", "final": "missing"}

        else:
            item["score"] = 0
            item["errorType"] = "unknown"

        if item["score"] >= 100:
            item["grade"] = "good"
        elif item["score"] >= 40:
            item["grade"] = "warn"
        else:
            item["grade"] = "error"

        word_analysis.append(item)

    return word_analysis, inserts, insert_feedbacks


def build_overall_rule_feedback(word_analysis: List[Dict[str, Any]], insert_feedbacks: List[str]) -> str:
    error_types = [x["errorType"] for x in word_analysis]

    has_delete = "delete" in error_types
    has_initial = any(t in error_types for t in ["initial", "initial+medial", "initial+final", "initial+medial+final"])
    has_medial = any(t in error_types for t in ["medial", "initial+medial", "medial+final", "initial+medial+final"])
    has_final = any(t in error_types for t in ["final", "initial+final", "medial+final", "initial+medial+final"])
    has_insert = bool(insert_feedbacks)

    if has_delete:
        return "일부 음절이 빠지거나 약하게 들린 부분이 있어요. 문장을 조금 천천히 말하면서 음절이 빠지지 않도록 해보세요."
    if has_initial:
        return "첫소리 구분이 흐려진 부분이 있어요. 단어를 시작할 때 첫소리를 조금 더 또렷하게 내보세요."
    if has_medial:
        return "모음이 다르게 전달된 부분이 있어요. 입 모양을 천천히 만들어 보면서 모음을 정확하게 발음해보세요."
    if has_final:
        return "받침이나 끝소리가 약하게 들린 부분이 있어요. 마지막 소리까지 힘을 빼지 말고 끝까지 유지해보세요."
    if has_insert:
        return "불필요한 소리가 덧붙은 부분이 있을 수 있어요. 급하게 말하기보다 일정한 속도로 또박또박 말해보세요."

    return "전반적으로 목표 문장과 비슷하게 잘 발화했어요. 지금처럼 천천히 또렷하게 말하는 연습을 계속해보세요."


def attach_syllable_timestamps(
    word_analysis: List[Dict[str, Any]],
    reference_text: str,
    whisperx_words: Optional[List[Dict[str, Any]]] = None
):
    if not whisperx_words:
        for item in word_analysis:
            item["start"] = None
            item["end"] = None
        return word_analysis

    ref_chars = split_korean_chars(reference_text)
    char_pointer = 0

    for item in word_analysis:
        item["start"] = None
        item["end"] = None

    for word_info in whisperx_words:
        word = str(word_info.get("word", "")).replace(" ", "")
        start = word_info.get("start")
        end = word_info.get("end")

        if not word or start is None or end is None:
            continue

        syllables = split_korean_chars(word)
        if not syllables:
            continue

        duration = max(0.0, float(end) - float(start))
        unit = duration / len(syllables) if len(syllables) > 0 else 0.0

        for i, _ in enumerate(syllables):
            if char_pointer >= len(ref_chars) or char_pointer >= len(word_analysis):
                break

            word_analysis[char_pointer]["start"] = round(float(start) + unit * i, 3)
            word_analysis[char_pointer]["end"] = round(float(start) + unit * (i + 1), 3)
            char_pointer += 1

    return word_analysis


def infer_reference_text(step_content: str, stt_text: str) -> str:
    """STT 결과(어눌한 발음)를 바탕으로 사용자가 말하려 했던 문장을 추정.
    시나리오 맥락은 힌트로만 사용하고, STT 텍스트를 최대한 유지하며 발음 오류만 교정."""
    system_prompt = """\
너는 언어 재활 평가 보조자야.
말이 어눌한 사용자의 STT 결과를 보고, 사용자가 실제로 말하려 했던 문장을 추정해.

규칙:
1. STT 결과를 최우선으로 유지해. 연습 맥락은 발음 교정 힌트로만 사용해.
2. 발음 오류로 인한 음절 치환·탈락만 교정해. 의미나 내용을 바꾸지 마.
   예: "아녕하세오" → "안녕하세요" (발음 교정 O)
   예: "아니요" → "다른 불편한 증상은 없습니다" (내용 변경 X, 절대 금지)
3. STT가 이미 종결 어미(요, 다, 까, 세요, 어요, 아요, 네요, 군요 등)로 끝나면 완결 문장으로 판단해.
   → 음절을 추가하지 말고, 발음 오류만 교정해. 음절 수를 STT와 동일하게 유지해.
4. STT가 명사·조사·어간 등 비종결 형태로 끝나면 미완결 문장으로 판단해.
   → 문장을 자연스럽게 끝맺는 종결어미까지만 보완 허용 (+3음절 이내).
   예: "아메리카노" → "아메리카노 주세요" (종결어 보완 O)
5. 반드시 JSON으로만 응답해.
{"inferredReferenceText": "..."}"""
    user_prompt = f"[연습 맥락 (힌트용)]\n{step_content}\n\n[STT 결과 (발음 교정 기준)]\n{stt_text}"
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("inferredReferenceText", "").strip()


def build_pronunciation_error_for_llm(
    word_analysis: List[Dict[str, Any]],
    insert_feedbacks: List[str],
) -> str:
    """LLM에 전달할 발음 오류 요약 텍스트 생성."""
    _TYPE_LABEL = {
        "delete":                 "음절 누락",
        "initial":                "초성 오류",
        "medial":                 "모음 오류",
        "final":                  "받침 오류",
        "initial+medial":         "초성·모음 오류",
        "initial+final":          "초성·받침 오류",
        "medial+final":           "모음·받침 오류",
        "initial+medial+final":   "초성·모음·받침 오류",
    }
    lines = []
    for item in word_analysis:
        if item["grade"] != "good":
            ref = item["refChar"]
            hyp = item["hypChar"] or "(누락)"
            label = _TYPE_LABEL.get(item.get("errorType", ""), "오류")
            lines.append(f'"{ref}" → "{hyp}" ({label})')
    if insert_feedbacks:
        lines.append(f"삽입 음절: {', '.join(insert_feedbacks)}")
    return "\n".join(lines) if lines else "오류 없음"


def generate_scenario_feedback(
    step_content: str,
    stt_text: str,
    pronunciation_error_summary: str,
    speech_rate: Dict[str, Any],
    pause_ratio: Dict[str, Any],
) -> Dict[str, Any]:
    """발음 오류·음성 분석·연습 맥락을 종합해 의미 전달률과 피드백을 JSON으로 반환."""
    system_prompt = """\
너는 성인 언어 재활 보조 평가자야.
아래 정보를 종합해서 의미 전달률과 피드백을 JSON으로만 반환해.

반환 형식:
{
  "meaningDeliveryScore": <0~100 정수>,
  "feedback": "<2~3문장>"
}

피드백 작성 규칙:
- 의미 전달 여부 + 발음 오류 + 조음 속도/pause 비율을 함께 고려해
- 잘한 점 먼저, 개선점은 구체적으로 뒤에
- 따뜻하고 격려하는 톤, 2~3문장으로 짧게
"""
    sps        = speech_rate.get("syllablesPerSecond", 0)
    sr_grade   = speech_rate.get("grade", "")
    sr_score   = speech_rate.get("score", 0)
    pause_pct  = pause_ratio.get("pausePercent", 0)
    pause_grade = pause_ratio.get("grade", "")

    user_prompt = f"""\
[연습 맥락]
{step_content}

[STT 결과]
{stt_text}

[발음 오류 분석]
{pronunciation_error_summary}

[조음 속도]
{sps} sps / 점수 {sr_score}/100 / 등급 {sr_grade}
(기준: 4.0–7.0 sps 정상 | slow: 느림 | fast: 빠름)

[침묵(Pause) 비율]
{pause_pct}% / 등급 {pause_grade}
(기준: ≤25% 정상 | warn: 쉬는 구간 많음 | error: 말 막힘 의심)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return json.loads(response.choices[0].message.content)


def generate_reference_feedback(
    reference_text: str,
    pronunciation_error_summary: str,
    speech_rate: Dict[str, Any],
    pause_ratio: Dict[str, Any],
) -> Dict[str, Any]:
    """발음 오류·음성 분석을 종합해 reference 연습용 피드백을 JSON으로 반환."""
    system_prompt = """\
너는 성인 언어 재활 보조 평가자야.
아래 정보를 종합해서 피드백을 JSON으로만 반환해.

반환 형식:
{
  "feedback": "<2~3문장>"
}

피드백 작성 규칙:
- 발음 오류 위치와 유형 + 조음 속도 + pause 비율을 함께 고려해
- 잘한 점 먼저, 개선점은 구체적으로 뒤에
- 따뜻하고 격려하는 톤, 2~3문장으로 짧게
"""
    sps         = speech_rate.get("syllablesPerSecond", 0)
    sr_grade    = speech_rate.get("grade", "")
    sr_score    = speech_rate.get("score", 0)
    pause_pct   = pause_ratio.get("pausePercent", 0)
    pause_grade = pause_ratio.get("grade", "")

    user_prompt = f"""\
[목표 문장]
{reference_text}

[발음 오류 분석]
{pronunciation_error_summary}

[조음 속도]
{sps} sps / 점수 {sr_score}/100 / 등급 {sr_grade}
(기준: 4.0–7.0 sps 정상 | slow: 느림 | fast: 빠름)

[침묵(Pause) 비율]
{pause_pct}% / 등급 {pause_grade}
(기준: ≤25% 정상 | warn: 쉬는 구간 많음 | error: 말 막힘 의심)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return json.loads(response.choices[0].message.content)


def generate_vowel_feedback(
    target_vowel: str,
    pronunciation: Dict[str, Any],
) -> Dict[str, Any]:
    """단모음 발음 정확도를 기반으로 AI 피드백을 JSON으로 반환."""
    system_prompt = """\
너는 성인 언어 재활 보조 평가자야.
단모음 연습 결과를 보고 피드백을 JSON으로만 반환해.

반환 형식:
{"feedback": "<2문장>"}

피드백 작성 규칙:
- 발음 정확도(포먼트) 기준으로 평가해
- 잘한 점 먼저, 개선점은 구체적으로 (입 모양, 혀 위치 등)
- 따뜻하고 격려하는 톤, 2문장으로 짧게
"""
    f1 = pronunciation.get("measuredF1")
    f2 = pronunciation.get("measuredF2")
    formant_info = (
        f"측정된 포먼트: F1={f1}Hz, F2={f2}Hz"
        if f1 is not None and f2 is not None
        else "포먼트 측정 불안정"
    )

    user_prompt = f"""\
[목표 모음]
{target_vowel}

[발음 정확도]
점수: {pronunciation.get("score", 0)}/100 / 등급: {pronunciation.get("grade", "")}
{formant_info}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return json.loads(response.choices[0].message.content)


def _get_acoustic_text(audio_path: Optional[str]) -> Optional[str]:
    """wav2vec2 음향 인식 시도. 불가 시 None 반환."""
    if not audio_path:
        return None
    try:
        from services.phoneme_acoustic_service import acoustic_recognize
        return acoustic_recognize(audio_path)
    except Exception:
        return None


def evaluate_reference_response(
    reference_text: str,
    stt_text: str,
    audio_path: Optional[str] = None,
    voice_result: Optional[Dict[str, Any]] = None,
    whisperx_words: Optional[List[Dict[str, Any]]] = None
):
    """
    참조 텍스트 기반 발음 평가.

    1. wav2vec2 음향 인식으로 발음 비교 (미설치 시 stt_text fallback)
    2. G2P 음소 분석으로 발음 점수·오류 산출
    3. LLM에 발음 오류 분석 + 음성 분석을 전달해 피드백 생성
    """
    reference_text = reference_text.strip()
    stt_text = stt_text.strip()

    # ── Step 1: G2P 발음 분석 ─────────────────────────────────────────────────
    acoustic_text = _get_acoustic_text(audio_path)
    comparison_text = acoustic_text if acoustic_text else stt_text

    word_analysis, inserts, insert_feedbacks = build_phoneme_based_analysis(
        reference_text, comparison_text
    )
    word_analysis = attach_syllable_timestamps(
        word_analysis=word_analysis,
        reference_text=reference_text,
        whisperx_words=whisperx_words
    )

    avg_word_score = round(
        sum(item["score"] for item in word_analysis) / len(word_analysis), 2
    ) if word_analysis else 0.0

    # ── Step 2: LLM 피드백 (발음 오류 + 음성 분석 포함) ─────────────────────
    pronunciation_error_summary = build_pronunciation_error_for_llm(word_analysis, insert_feedbacks)
    speech_rate = (voice_result or {}).get("speechRate", {})
    pause_ratio = (voice_result or {}).get("silenceRatio", {})

    llm_result = generate_reference_feedback(
        reference_text=reference_text,
        pronunciation_error_summary=pronunciation_error_summary,
        speech_rate=speech_rate,
        pause_ratio=pause_ratio,
    )
    feedback = llm_result.get("feedback", "").strip()

    simplified_word_analysis = [
        {"refChar": item["refChar"], "hypChar": item["hypChar"], "grade": item["grade"]}
        for item in word_analysis
    ]

    return {
        "referenceText": reference_text,
        "sttText": stt_text,
        "acousticText": acoustic_text,
        "pronunciationScore": round(float(avg_word_score), 2),
        "feedback": feedback,
        "wordAnalysis": simplified_word_analysis,
    }


def evaluate_scenario_response(
    step_content: str,
    stt_text: str,
    audio_path: Optional[str] = None,
    voice_result: Optional[Dict[str, Any]] = None,
    whisperx_words: Optional[List[Dict[str, Any]]] = None
):
    """
    시나리오 맥락 기반 발음 평가.

    1. LLM으로 참조 텍스트 추론
    2. G2P 음소 분석으로 발음 점수·오류 산출
    3. LLM에 발음 오류 분석 + 연습 맥락 + 음성 분석을 전달해 피드백 생성
    """
    # ── Step 1: 참조 텍스트 추론 ───────────────────────────────────────────────
    reference_text = infer_reference_text(step_content, stt_text)

    # ── Step 2: G2P 발음 분석 ─────────────────────────────────────────────────
    acoustic_text = _get_acoustic_text(audio_path)
    comparison_text = acoustic_text if acoustic_text else stt_text

    word_analysis, inserts, insert_feedbacks = build_phoneme_based_analysis(
        reference_text, comparison_text
    )
    word_analysis = attach_syllable_timestamps(
        word_analysis=word_analysis,
        reference_text=reference_text,
        whisperx_words=whisperx_words
    )

    avg_word_score = round(
        sum(item["score"] for item in word_analysis) / len(word_analysis), 2
    ) if word_analysis else 0.0

    # ── Step 3: LLM 피드백 (발음 오류 + 음성 분석 포함) ─────────────────────
    pronunciation_error_summary = build_pronunciation_error_for_llm(word_analysis, insert_feedbacks)
    speech_rate = (voice_result or {}).get("speechRate", {})
    pause_ratio = (voice_result or {}).get("silenceRatio", {})

    llm_result = generate_scenario_feedback(
        step_content=step_content,
        stt_text=stt_text,
        pronunciation_error_summary=pronunciation_error_summary,
        speech_rate=speech_rate,
        pause_ratio=pause_ratio,
    )

    meaning_delivery_score = clamp(safe_int(llm_result.get("meaningDeliveryScore", 0), 0), 0, 100)
    feedback = llm_result.get("feedback", "").strip()

    simplified_word_analysis = [
        {"refChar": item["refChar"], "hypChar": item["hypChar"], "grade": item["grade"]}
        for item in word_analysis
    ]

    return {
        "referenceText": reference_text,
        "sttText": stt_text,
        "acousticText": acoustic_text,
        "pronunciationScore": round(float(avg_word_score), 2),
        "meaningDeliveryScore": meaning_delivery_score,
        "feedback": feedback,
        "wordAnalysis": simplified_word_analysis,
    }
