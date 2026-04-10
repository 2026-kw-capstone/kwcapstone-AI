from difflib import SequenceMatcher
from jiwer import wer, cer
from openai import OpenAI
from config.settings import OPENAI_API_KEY
import json
from typing import List, Dict, Any, Optional

client = OpenAI(api_key=OPENAI_API_KEY)


def split_korean_chars(text: str) -> List[str]:
    return [ch for ch in text.strip() if ch.strip()]


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


def evaluate_with_llm(step_content: str, stt_text: str):
    system_prompt = """
너는 성인 언어 재활 보조 평가자임.

사용자의 발화(STT 결과)를 보고 아래만 평가해야 함.
1. 사용자가 말하려던 의도 문장 추정
2. step 내용과 비교했을 때 의미 전달률(0~100)
3. 전체 발화에 대한 짧은 피드백

반드시 JSON만 반환해야 함.

반환 형식:
{
  "inferredReferenceText": "...",
  "meaningDeliveryScore": 0,
  "feedback": "..."
}
"""

    user_prompt = f"""
[원래 연습 목표 문장/상황]
{step_content}

[사용자 STT 결과]
{stt_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)


def evaluate_reference_response(
    reference_text: str,
    stt_text: str,
    whisperx_words: Optional[List[Dict[str, Any]]] = None
):
    reference_text = reference_text.strip()
    stt_text = stt_text.strip()

    reference_scores = calculate_reference_scores(reference_text, stt_text)
    alignment_result = build_alignment(reference_text, stt_text)

    word_analysis, inserts, insert_feedbacks = build_rule_based_analysis(
        reference_text, alignment_result
    )

    word_analysis = attach_syllable_timestamps(
        word_analysis=word_analysis,
        reference_text=reference_text,
        whisperx_words=whisperx_words
    )

    rule_feedback = build_overall_rule_feedback(word_analysis, insert_feedbacks)

    avg_word_score = round(
        sum(item["score"] for item in word_analysis) / len(word_analysis), 2
    ) if word_analysis else 0.0

    simplified_word_analysis = [
        {"refChar": item["refChar"], "hypChar": item["hypChar"], "grade": item["grade"]}
        for item in word_analysis
    ]

    return {
        "referenceText": reference_text,
        "sttText": stt_text,
        "pronunciationScore": round(float(avg_word_score), 2),
        "meaningDeliveryScore": round(max(0.0, 100 * (1 - float(reference_scores.get("cer", 1.0)))), 2),
        "feedback": rule_feedback,
        "wordAnalysis": simplified_word_analysis,
    }


def evaluate_scenario_response(
    step_content: str,
    stt_text: str,
    whisperx_words: Optional[List[Dict[str, Any]]] = None
):
    llm_result = evaluate_with_llm(step_content, stt_text)

    reference_text = llm_result.get("inferredReferenceText", "").strip()
    meaning_delivery_score = clamp(safe_int(llm_result.get("meaningDeliveryScore", 0), 0), 0, 100)
    llm_feedback = llm_result.get("feedback", "").strip()

    alignment_result = build_alignment(reference_text, stt_text)

    word_analysis, inserts, insert_feedbacks = build_rule_based_analysis(
        reference_text, alignment_result
    )

    word_analysis = attach_syllable_timestamps(
        word_analysis=word_analysis,
        reference_text=reference_text,
        whisperx_words=whisperx_words
    )

    rule_feedback = build_overall_rule_feedback(word_analysis, insert_feedbacks)

    avg_word_score = round(
        sum(item["score"] for item in word_analysis) / len(word_analysis), 2
    ) if word_analysis else 0.0

    final_feedback = f"{llm_feedback} {rule_feedback}".strip() if llm_feedback else rule_feedback

    simplified_word_analysis = [
        {"refChar": item["refChar"], "hypChar": item["hypChar"], "grade": item["grade"]}
        for item in word_analysis
    ]

    return {
        "referenceText": reference_text,
        "sttText": stt_text,
        "pronunciationScore": round(float(avg_word_score), 2),
        "meaningDeliveryScore": meaning_delivery_score,
        "feedback": final_feedback,
        "wordAnalysis": simplified_word_analysis,
    }
