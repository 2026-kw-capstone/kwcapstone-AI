# 🎙️ AI Speech Pipeline for Communication Rehabilitation

성인 의사소통 및 조음 장애 사용자의 재활을 돕기 위한 **AI 기반 언어 훈련 파이프라인**입니다.
OpenAI Whisper STT, GPT-4o-mini LLM, TTS를 결합하여 시나리오 생성, 정교한 발음 평가, 음성 품질 분석, 단모음 발음 정확도 분석, 격려 기반 자유 대화 기능을 제공합니다.

---

## ✨ 주요 기능 (Key Features)

### 1. 맞춤형 훈련 시나리오 생성 (`scenario_service`)
- 사용자의 장소(병원, 카페 등)와 목적에 맞춰 **3단계 난이도(Level 1~3)** 시나리오를 자동 생성합니다.
- 각 레벨은 3개의 스텝으로 구성되어 자연스러운 대화 흐름을 유도합니다.
- 각 스텝은 `assistantMessage`(AI 질문)와 `userIntent`(사용자 연습 목표)로 구성됩니다.

### 2. 음소 단위 발음 분석 및 피드백 (`score_service`, `phoneme_acoustic_service`)
- **wav2vec2 CTC 음향 인식**으로 발음 점수를 산출합니다. Whisper의 언어 모델 보정을 우회하여 실제 발화된 음소를 그대로 반영합니다.
  - **모음 왜곡** 감지: "아" 대신 "어"로 발음 → Whisper는 문맥으로 보정하지만 wav2vec2는 "어"로 출력
  - **받침 약화** 감지: "약"을 "야"로 발음 → Whisper가 받침을 채워 넣지만 wav2vec2는 누락 그대로 출력
  - **기식음 혼동** 감지: ㅂ/ㅍ 구분 등 음향적 미세 차이 감지
  - 모델(`kresnik/wav2vec2-large-xlsr-korean`) 미설치 또는 인식 실패 시 Whisper STT로 자동 fallback
- **G2P(Grapheme-to-Phoneme)** 변환으로 한국어 음운 규칙(연음화·경음화·비음화·격음화)을 추가 적용합니다.
  - 예) `닭이` → `달기`, `학교` → `학꾜`, `국민` → `궁민`
  - 표기가 달라도 발음이 같으면 정확한 발음으로 인정합니다.
- 각 음절을 **초성/중성/종성 음소 단위**로 분해하여 flat 시퀀스를 만들고, SequenceMatcher로 음소 단위 정렬을 수행합니다.
- 음소 오류를 다시 원본 음절 단위로 집계하여 `grade`(good / warn / error)를 부여, UI 색상 렌더링에 활용합니다.
- **Reference 모드**: wav2vec2 음향 인식 결과와 목표 문장을 G2P 변환 후 음소 단위로 비교합니다.
- **Scenario 모드**: LLM이 시나리오 맥락으로 의도 문장을 추정한 뒤, wav2vec2 결과와 동일한 G2P 음소 분석을 적용합니다.

### 3. 음성 품질 분석 (`voice_analysis_service`)
- 실제 오디오 파일을 librosa로 분석하여 두 가지 음성 품질 지표를 제공합니다.
- **조음 속도**: pause를 제거한 발화 시간 기준으로 음절/초를 산출합니다. (정상 범위: 4.0~7.0 sps, Lee et al. 2017; ASHA 2003)
- **침묵 비율**: 임상적 유의 pause(≥ 250ms)만 집계하여 말 막힘 및 과도한 쉼을 감지합니다. (Angelopoulou et al., Brain Sciences 2024)

### 4. 음성 합성 TTS (`tts_service`)
- OpenAI `tts-1` 모델(기본 음성: nova)을 사용해 목표 문장을 음성으로 변환합니다.
- 사용자가 올바른 발음을 귀로 먼저 확인할 수 있도록 돕습니다.

### 5. 지능형 음성 인식 (`stt_service`)
- **OpenAI Whisper** 모델을 사용하여 불분명한 발음도 높은 정확도로 텍스트화합니다.
  - 로컬(PyCharm/VSCode): `small` 모델
  - Colab: `large` 모델
- `pydub`를 활용해 오디오를 16kHz Mono WAV로 전처리하여 인식률을 최적화합니다.

### 6. 격려 기반 AI 자유 대화 (`chat_service`)
- 언어 장애 사용자를 배려한 따뜻한 말투의 대화 파트너 기능을 제공합니다.
- 교정보다는 **자신감 향상**에 초점을 맞춘 응답 시스템입니다.

### 7. 단모음 발음 정확도 분석 (`voice_analysis_service`)
- 한국어 단모음(`아`, `이`, `우`, `에`, `오`, `애`, `외`, `위`, `으`, `의`) 발음의 정확도를 **LPC 포먼트 분석**으로 평가합니다.
- 가장 긴 유성 구간의 중간 60%를 안정 구간으로 추출하여 F1·F2 포먼트를 측정합니다.
- 한국어 단모음 기준 포먼트와 비교하여 0~100점 산출합니다. (`score = max(0, 100 − |ΔF1|/3) × 0.5 + max(0, 100 − |ΔF2|/6) × 0.5`)

---

## 📂 프로젝트 구조 (Project Structure)

```
ai_speech_pipeline/
│
├── .env                         # API 키 설정 (git 제외)
├── .env.example                 # 환경 변수 예시 파일
├── requirements.txt             # 의존성 패키지 목록
├── main.py                      # 서버 실행 진입점
├── test_analysis.py             # 발음 분석 테스트 스크립트 (오디오 없이 실행 가능)
│
├── config/
│   └── settings.py              # 환경 변수 로드
│
├── services/
│   ├── scenario_service.py       # 시나리오 생성 로직 (GPT-4o-mini)
│   ├── audio_service.py          # S3 다운로드 및 오디오 전처리
│   ├── stt_service.py            # Whisper 기반 음성 인식 (의미 이해용)
│   ├── phoneme_acoustic_service.py # wav2vec2 CTC 음향 인식 (발음 점수용)
│   ├── score_service.py          # G2P + 음소 시퀀스 정렬 기반 발음 평가
│   ├── voice_analysis_service.py # 조음속도/침묵비율 음성 품질 분석 + 단모음 LPC 포먼트 분석
│   ├── tts_service.py            # OpenAI TTS 음성 합성
│   └── chat_service.py           # LLM 기반 자유 대화
│
├── api/
│   └── app.py                   # FastAPI 엔드포인트 정의
│
└── data/
    ├── input/                   # 수신된 원본 오디오 저장
    └── output/                  # 전처리된 오디오 저장
```

---

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정

`.env.example`을 복사하여 `.env` 파일을 생성하고 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일 내용:

```
OPENAI_API_KEY=your_openai_api_key_here
COLAB_API_TOKEN=your_token_here
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
# ffmpeg가 없는 경우 (오디오 전처리에 필요)
# macOS: brew install ffmpeg
# Ubuntu: apt-get install -y ffmpeg
```

> **macOS에서 g2pk(G2P) 설치 시 주의**
> g2pk는 Java(JDK)와 `ninja`, `ant` 빌드 도구가 필요합니다.
> ```bash
> brew install ninja ant
> # JDK 설치 후 JAVA_HOME을 지정해서 pip install 실행
> JAVA_HOME="$(/usr/libexec/java_home)" pip install -r requirements.txt
> ```

### 3. 서버 실행

```bash
python main.py
```

서버가 실행되면 `http://localhost:8000/docs`에서 Swagger UI로 API를 테스트할 수 있습니다.

### 4. 발음 분석 테스트 (오디오 파일 없이)

```bash
python test_analysis.py             # reference + scenario 모두 실행
python test_analysis.py --mode reference
python test_analysis.py --mode scenario
```

---

## 🛠️ API 명세 (API Endpoints)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/stt` | 오디오(S3 URL)를 텍스트로 변환 |
| `POST` | `/tts` | 텍스트를 음성(MP3)으로 변환 |
| `POST` | `/generate-scenario` | 상황/목적에 따른 3단계 훈련 시나리오 생성 |
| `POST` | `/practice/reference` | 목표 문장 기반 발음 정밀 분석 + 음성 품질 분석 |
| `POST` | `/practice/scenario` | 시나리오 맥락 내 LLM 의도 추정 후 발음 평가 + 음성 품질 분석 |
| `POST` | `/practice/vowel` | 단모음 LPC 포먼트 기반 발음 정확도 분석 |
| `POST` | `/chat/free-talk` | AI 대화 파트너와의 격려형 자유 대화 |

---

## 🧮 핵심 알고리즘: G2P 기반 음소 단위 발음 분석

### 처리 흐름

```
참조 텍스트 ──→ G2P(음운 규칙) ──→ 음소 flat 시퀀스 ──┐
                                                        ├─→ 음소 단위 정렬 → 음절별 오류 집계 → 점수/등급
STT 텍스트  ──→ G2P(음운 규칙) ──→ 음소 flat 시퀀스 ──┘
```

### 적용 음운 규칙 (g2pk)

| 규칙 | 예시 | 변환 결과 |
|------|------|-----------|
| 연음화 | 닭이 | 달기 |
| 경음화 | 학교 | 학꾜 |
| 비음화 | 국민 | 궁민 |
| 격음화 | 좋다 | 조타 |

> 표기가 달라도 G2P 변환 결과가 동일하면 **정확한 발음으로 인정**합니다.

### 발음 점수 산출

WER은 띄어쓰기 생략 시 점수가 왜곡되므로, G2P 변환 후 **음소 단위 오류를 음절별로 집계한 평균 점수**를 사용합니다.

$$\text{pronunciationScore} = \frac{1}{N} \sum_{i=1}^{N} \text{syllableScore}_i$$

### 음절 점수 기준

각 음절의 점수는 G2P 변환 후 음소 정렬에서 감지된 오류 음소(초성/중성/종성) 개수로 결정됩니다.

| 오류 음소 수 | 점수 | 등급 |
|------------|------|------|
| 0개 (정확) | 100 | good |
| 1개 | 70 | warn |
| 2개 | 40 | warn |
| 3개 이상 | 10 | error |
| 음절 누락 | 0 | error |

### Grade 기준 (UI 색상 렌더링용)

| Grade | 조건 | 의미 |
|-------|------|------|
| `good` | score = 100 | 정확한 발음 |
| `warn` | score ≥ 40 | 부분 오류 |
| `error` | score < 40 | 심각한 오류 또는 누락 |

### 의미 전달 점수 (meaningDeliveryScore)

- **Scenario 모드 전용**: GPT-4o-mini가 시나리오 맥락과 STT 결과를 비교하여 0~100 평가

### 음성 품질 분석 기준 (`/practice/reference`, `/practice/scenario`)

| 항목 | 측정 방식 | good | warn | error |
|------|----------|------|------|-------|
| 조음 속도 | pause 제거 후 음절/초 | 4.0 ~ 7.0 sps | 4.0 미만 또는 7.0 초과 | 0 또는 9.0+ |
| 침묵 비율 | ≥ 250ms pause 점유율 | ≤ 25% | 25 ~ 45% | > 45% |

### 단모음 발음 정확도 기준 (`/practice/vowel`)

| 점수 | 등급 | 의미 |
|------|------|------|
| ≥ 75 | good | 모음 발음이 정확해요 |
| 50 ~ 74 | warn | 모음 발음이 조금 어긋났어요 |
| < 50 | error | 모음 발음을 다시 연습해보세요 |

---

## 📋 기술 스택 (Tech Stack)

- **Backend**: FastAPI, Uvicorn
- **AI/ML**: OpenAI GPT-4o-mini, Whisper (small/large), OpenAI TTS (tts-1)
- **Audio Processing**: Pydub, FFmpeg, librosa
- **Phoneme Analysis**: g2pk (Korean G2P), konlpy, jamo
- **Evaluation**: JiWER (WER/CER), Difflib (SequenceMatcher)
- **Environment**: python-dotenv, Python 3.9+
- **Build Tools** (macOS): ninja, ant (g2pk 의존성)
