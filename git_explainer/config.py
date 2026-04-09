import os

from dotenv import load_dotenv

load_dotenv()

# --- Required (fail fast if missing) ---
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
KIMI_API_KEY = os.environ["KIMI_API_KEY"]

# --- Kimi API settings ---
KIMI_BASE_URL = "https://api.moonshot.cn/v1"
KIMI_MODEL = os.getenv("KIMI_MODEL", "moonshot-v1-8k")
KIMI_MAX_TOKENS = int(os.getenv("KIMI_MAX_TOKENS", "4096"))

# --- GitHub API ---
GITHUB_API_BASE = "https://api.github.com"
