import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLAB_API_TOKEN = os.getenv("COLAB_API_TOKEN")
