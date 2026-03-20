# 🎙️ AI Speech Pipeline for Communication Rehabilitation

이 프로젝트는 성인 의사소통 및 조음 장애 사용자의 재활을 돕기 위한 **AI 기반 언어 훈련 파이프라인**입니다. OpenAI의 LLM과 Whisper STT를 결합하여 실시간 시나리오 생성, 정교한 발음 평가 및 피드백, 그리고 따뜻한 격려 기반의 자유 대화 기능을 제공합니다.

---

## ✨ 주요 기능 (Key Features)

## 1. 맞춤형 훈련 시나리오 생성 (`Scenario Service`)

- 사용자의 상황과 목적에 맞춰 **3단계 난이도(Level 1~3)**의 시나리오를 자동 생성합니다.
- 각 레벨은 3가지 스텝으로 구성되어 자연스러운 대화 흐름을 유도합니다.

## 2. 정교한 발음 분석 및 피드백 (`Score Service`)

- **WER(Word Error Rate)** 및 **CER(Character Error Rate)** 기반의 정량적 평가를 제공합니다.
- *한글 자모 분해(Initial/Medial/Final)**를 통해 "초성이 흐림", "종성이 약함"과 같은 구체적인 조음 피드백을 생성합니다.
- 문장 끝부분 처리 및 중간 음절 치환 등 언어 습관에 대한 규칙 기반 분석을 수행합니다.

## 3. 지능형 음성 인식 (`STT Service`)

- **OpenAI Whisper (Large)** 모델을 사용하여 노이즈가 있거나 불분명한 발음도 높은 정확도로 텍스트화합니다.
- `pydub`를 활용해 오디오를 16kHz Mono WAV 포맷으로 전처리하여 인식률을 최적화합니다.

## 4. 격려 기반 AI 자유 대화 (`Chat Service`)

- 언어 장애 사용자를 배려한 따뜻한 말투의 대화 파트너 기능을 제공합니다.
- 교정보다는 **자신감 향상**에 초점을 맞춘 응답 시스템입니다.

---

## 📂 프로젝트 구조 (Project Structure)

```
ai_speech_pipeline/
│
├── .env                # API 키 및 설정 (OpenAI, Ngrok 등)
├── requirements.txt    # 의존성 패키지 목록
│
├── config/
│   └── settings.py     # 환경 변수 및 설정 로드
│
├── services/
│   ├── scenario_service.py # 시나리오 생성 로직
│   ├── audio_service.py    # S3 다운로드 및 오디오 전처리
│   ├── stt_service.py      # Whisper 기반 음성 인식
│   ├── score_service.py    # 발음 평가 및 자모 분석 알고리즘
│   └── chat_service.py     # LLM 기반 자유 대화
│
├── api/
│   └── app.py              # FastAPI 엔드포인트 정의
│
└── data/
    ├── input/              # 수신된 원본 오디오 저장
    └── output/             # 전처리된 오디오 저장
```

---

## 🚀 시작하기 (Getting Started)

## 1. 환경 설정

프로젝트 루트에 `.env` 파일을 생성하거나 Google Colab 환경 변수에 다음 정보를 설정해야 합니다.

- `OPENAI_API_KEY`: OpenAI 서비스 이용을 위한 키
- `NGROK_AUTH_TOKEN`: 외부 접근을 위한 Ngrok 토큰 (선택 사항)
- `COLAB_API_TOKEN`: API 보안을 위한 인증 토큰

## 2. 패키지 설치

Bash

`pip install -r requirements.txt
apt-get install -y ffmpeg`

## 3. 서버 실행

Python

`uvicorn.run(app, host="0.0.0.0", port=8000)`

---

## 🛠️ API 명세 (API Endpoints)

| **Method** | **Endpoint** | **Description** |
| --- | --- | --- |
| `POST` | `/generate-scenario` | 상황과 목적에 따른 3단계 훈련 시나리오 생성 |
| `POST` | `/practice/reference` | 특정 목표 문장을 따라 읽은 발음 정밀 분석 |
| `POST` | `/practice/scenario` | 시나리오 맥락 내에서 LLM이 의도 문장을 추정하여 평가 |
| `POST` | `/chat/free-talk` | AI 대화 파트너와의 격려형 자유 대화 |
| `GET` | `/health` | 서버 상태 확인 |

---

## 🧮 핵심 알고리즘: 한글 조음 분석

이 시스템은 단순히 텍스트 일치 여부만 확인하지 않습니다. 한글의 특성을 고려하여 다음과 같은 수식을 통해 발음 점수를 산출합니다.

- **발음 점수 ($Pronunciation Score$):**
    
    $$Score = \frac{100 \times (1 - WER) + Avg(Syllable Score)}{2}$$
    
- **자모 분해 분석:** 각 음절을 초성, 중성, 종성으로 분리하여 어느 부분에서 발음 오류가 발생했는지 감지하고 맞춤형 가이드를 제공합니다.

---

## 📋 기술 스택 (Tech Stack)

- **Backend**: FastAPI, Uvicorn
- **AI/ML**: OpenAI GPT-4o/GPT-5, Whisper
- **Audio Processing**: Pydub, FFmpeg
- **Evaluation**: JiWER (WER/CER), Difflib
- **Deployment**: Ngrok (Colab Environment)
