"""Settings, read from the environment (.env).

Falls back to the old .streamlit/secrets.toml for the auth/Supabase values so
an existing local setup keeps working without re-entering secrets.
"""
import os
import tomllib
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_legacy: dict = {}
_legacy_path = ROOT / ".streamlit" / "secrets.toml"
if _legacy_path.exists():
    with open(_legacy_path, "rb") as f:
        _legacy = tomllib.load(f)


def _get(env_name: str, legacy_section: str, legacy_key: str, default: str = "") -> str:
    return os.getenv(env_name) or _legacy.get(legacy_section, {}).get(legacy_key) or default


GOOGLE_CLIENT_ID = _get("GOOGLE_CLIENT_ID", "auth", "client_id")
GOOGLE_CLIENT_SECRET = _get("GOOGLE_CLIENT_SECRET", "auth", "client_secret")
SESSION_SECRET = _get("SESSION_SECRET", "auth", "cookie_secret")
SUPABASE_URL = _get("SUPABASE_URL", "supabase", "url")
SUPABASE_SERVICE_KEY = _get("SUPABASE_SERVICE_KEY", "supabase", "service_key")

# Vercel sets VERCEL=1 in its build and runtime environments.
ON_VERCEL = os.getenv("VERCEL") == "1"

# Optional override for the OAuth callback URL. By default it's derived from
# the incoming request's host (see backend/main.py), so the same code works on
# localhost:5173 and on the Vercel domain. Either way, the URL must exactly
# match an "Authorized redirect URI" in Google Cloud Console.
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "")

# Session cookie only sent over HTTPS — on by default in production.
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "1" if ON_VERCEL else "0") == "1"

AUTH_CONFIGURED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and SESSION_SECRET)
