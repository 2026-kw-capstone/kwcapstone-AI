import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLAB_API_TOKEN = os.getenv("COLAB_API_TOKEN")

# Redis 세션 저장소 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# 세션 만료 시간(초). 기본 6시간.
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(6 * 60 * 60)))
