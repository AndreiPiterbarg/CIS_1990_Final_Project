import os

from dotenv import load_dotenv

load_dotenv()

# --- Required (fail fast if missing) ---
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# --- Groq API settings (OpenAI-compatible endpoint) ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "4096"))

# --- GitHub API ---
GITHUB_API_BASE = "https://api.github.com"
