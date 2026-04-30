# 🎙️ AI Speech Pipeline for Communication Rehabilitation

성인 의사소통 및 조음 장애 사용자의 재활을 돕기 위한 **AI 기반 언어 훈련 파이프라인**입니다.
OpenAI Whisper STT, GPT-4o-mini LLM, TTS를 결합하여 시나리오 생성, 정교한 발음 평가, 음성 품질 분석, 격려 기반 자유 대화 기능을 제공합니다.

---

## ✨ 주요 기능 (Key Features)

### 1. 맞춤형 훈련 시나리오 생성 (`scenario_service`)
- 사용자의 장소(병원, 카페 등)와 목적에 맞춰 **3단계 난이도(Level 1~3)** 시나리오를 자동 생성합니다.
- 각 레벨은 3개의 스텝으로 구성되어 자연스러운 대화 흐름을 유도합니다.
- 각 스텝은 `assistantMessage`(AI 질문)와 `userIntent`(사용자 연습 목표)로 구성됩니다.

### 2. 정교한 발음 분석 및 피드백 (`score_service`)
- **한글 자모 분해(초성/중성/종성)** 기반으로 음절별 오류 위치를 정밀하게 분석합니다.
- SequenceMatcher 정렬을 통해 equal / substitute / delete / insert 오류 유형을 구분합니다.
- 각 음절에 `grade`(good / warn / error)를 부여하여 UI 색상 렌더링에 활용합니다.
- **Reference 모드**: 사용자가 입력한 목표 문장과 STT 결과를 직접 비교합니다.
- **Scenario 모드**: LLM이 STT 결과와 시나리오 맥락을 보고 의도 문장을 추정한 뒤 평가합니다.

### 3. 음성 품질 분석 (`voice_analysis_service`)
- 실제 오디오 파일을 librosa로 분석하여 세 가지 음성 품질 지표를 제공합니다.
- **음량**: RMS 에너지를 dBFS로 변환하여 목소리 크기의 적절성을 평가합니다.
- **발화 속도**: 한글 음절 수를 발화 시간으로 나눠 음절/초를 산출합니다. (정상 범위: 3~6음절/초)
- **침묵 비율**: 무음 구간 비율로 말 막힘 및 과도한 쉼을 감지합니다.

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

### 7. 한 음절 발성 훈련 분석 (`voice_analysis_service`)
- 한 음절 오디오를 분석하여 **음량**과 **발성 시간**을 각각 0~100점으로 산출합니다.
- **음량 점수**: RMS 에너지 → dBFS 변환 후 선형 매핑 (-15 dBFS = 100점, -45 dBFS = 0점)
- **발성 시간 점수**: 무음 구간 제거 후 실제 발성 시간 측정 (0~0.3초: 0~50점 / 0.3~2.0초: 50~100점 / 2.0초 이상: 100점)

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
│   ├── scenario_service.py      # 시나리오 생성 로직 (GPT-4o-mini)
│   ├── audio_service.py         # S3 다운로드 및 오디오 전처리
│   ├── stt_service.py           # Whisper 기반 음성 인식
│   ├── score_service.py         # 발음 평가 및 한글 자모 분석
│   ├── voice_analysis_service.py# 음량/발화속도/침묵비율 음성 품질 분석
│   ├── tts_service.py           # OpenAI TTS 음성 합성
│   └── chat_service.py          # LLM 기반 자유 대화
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
| `POST` | `/practice/syllable-voice` | 한 음절 오디오의 음량 + 발성 시간 점수 측정 |
| `POST` | `/chat/free-talk` | AI 대화 파트너와의 격려형 자유 대화 |

---

## 🧮 핵심 알고리즘: 한글 조음 분석

### 발음 점수 산출

WER은 STT가 띄어쓰기를 생략하면 점수가 왜곡되는 문제가 있어, **음절 단위 평균 점수**만을 사용합니다.

$$\text{pronunciationScore} = \frac{1}{N} \sum_{i=1}^{N} \text{syllableScore}_i$$

### 음절 점수 기준

각 음절은 자모(초성/중성/종성) 차이 개수에 따라 점수가 부여됩니다.

| 오류 유형 | 다른 자모 수 | 점수 |
|----------|------------|------|
| 정확 (equal) | 0 | 100 |
| 경미한 오류 | 1 | 70 |
| 중간 오류 | 2 | 40 |
| 심각한 오류 | 3 | 10 |
| 누락 (delete) | - | 0 |

### Grade 기준 (UI 색상 렌더링용)

| Grade | 조건 | 의미 |
|-------|------|------|
| `good` | score ≥ 100 | 정확한 발음 |
| `warn` | score ≥ 40 | 부분 오류 |
| `error` | score < 40 | 심각한 오류 또는 누락 |

### 의미 전달 점수 (meaningDeliveryScore)

- **Reference 모드**: CER 기반 → `max(0, 100 × (1 - CER))`
- **Scenario 모드**: GPT-4o-mini가 시나리오 맥락과 STT 결과를 비교하여 0~100 평가

### 음성 품질 분석 기준 (`/practice/reference`, `/practice/scenario`)

| 항목 | good | warn | error |
|------|------|------|-------|
| 음량 | > -25 dBFS | -35 ~ -25 | < -35 dBFS |
| 발화 속도 | 3.0 ~ 6.0 음절/초 | 2.0 ~ 3.0 또는 6.0 ~ 7.5 | < 2.0 또는 > 7.5 |
| 침묵 비율 | ≤ 25% | 25 ~ 45% | > 45% |

### 한 음절 발성 분석 점수 기준 (`/practice/syllable-voice`)

| 항목 | 측정 방식 | 100점 기준 |
|------|----------|-----------|
| 음량 | RMS → dBFS 선형 매핑 | -15 dBFS 이상 |
| 발성 시간 | 무음 제거 후 발성 구간 합산 | 2.0초 이상 |

등급은 두 항목 공통으로 `score ≥ 75` → good / `40~74` → warn / `< 40` → error 입니다.

---

## 📋 기술 스택 (Tech Stack)

- **Backend**: FastAPI, Uvicorn
- **AI/ML**: OpenAI GPT-4o-mini, Whisper (small/large), OpenAI TTS (tts-1)
- **Audio Processing**: Pydub, FFmpeg, librosa
- **Evaluation**: JiWER (WER/CER), Difflib (SequenceMatcher)
- **Environment**: python-dotenv, Python 3.9+
