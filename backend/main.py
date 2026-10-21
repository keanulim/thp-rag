"""FastAPI server for the Dunk Bot React frontend.

Run: .venv/bin/uvicorn backend.main:app --reload --port 8000
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from . import config, rag, store

app = FastAPI(title="Dunk Bot API")

if config.AUTH_CONFIGURED:
    # Signed (not encrypted) cookie holding just {email, name}. SameSite=Lax
    # means browsers won't attach it to cross-site POST/DELETEs, which is what
    # protects the mutating endpoints below from CSRF.
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.SESSION_SECRET,
        same_site="lax",
        https_only=config.SESSION_HTTPS_ONLY,
        max_age=60 * 60 * 24 * 30,
    )

oauth = OAuth()
oauth.register(
    name="google",
    client_id=config.GOOGLE_CLIENT_ID,
    client_secret=config.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

_title_pool = ThreadPoolExecutor(max_workers=4)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dunkbot")


def _log_error(context: str, exc: Exception):
    # stdout/stderr, not a file: Vercel's filesystem is read-only, and its
    # dashboard (Logs tab) shows whatever the function prints.
    log.error("[%s] %r", context, exc)


def current_user(request: Request) -> dict | None:
    if not config.AUTH_CONFIGURED:
        return None
    return request.session.get("user")


def require_email(request: Request) -> str:
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Log in to use saved chats.")
    return user["email"]


# --- Auth ---

def _redirect_uri(request: Request) -> str:
    if config.OAUTH_REDIRECT_URI:
        return config.OAUTH_REDIRECT_URI
    # Behind Vercel's proxy (and Vite's in dev) the public host and scheme
    # arrive in X-Forwarded-* / Host headers. A spoofed Host can't hijack
    # login: Google refuses any redirect URI that isn't registered.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", request.url.netloc)
    return f"{proto}://{host}/api/auth/callback"


@app.get("/api/auth/login")
async def login(request: Request):
    if not config.AUTH_CONFIGURED:
        raise HTTPException(503, "Google login isn't configured on the server.")
    return await oauth.google.authorize_redirect(request, _redirect_uri(request))


@app.get("/api/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        _log_error("auth_callback", e)
        return RedirectResponse("/?login_error=1")
    info = token["userinfo"]
    request.session["user"] = {
        "email": info["email"],
        "name": info.get("given_name") or info.get("name") or info["email"].split("@")[0],
    }
    return RedirectResponse("/")


@app.post("/api/auth/logout")
def logout(request: Request):
    if config.AUTH_CONFIGURED:
        request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    return {"user": current_user(request), "auth_configured": config.AUTH_CONFIGURED}


# --- Saved chats ---

@app.get("/api/chats")
def get_chats(request: Request):
    email = require_email(request)
    try:
        return {"chats": store.list_chats(email)}
    except Exception as e:
        _log_error("list_chats", e)
        raise HTTPException(502, "Couldn't load saved chats — Supabase is unreachable.")


@app.get("/api/chats/{chat_id}/messages")
def get_chat_messages(chat_id: str, request: Request):
    email = require_email(request)
    try:
        return {"messages": store.get_messages(email, chat_id)}
    except Exception as e:
        _log_error("get_messages", e)
        raise HTTPException(502, "Couldn't load this chat — Supabase is unreachable.")


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, request: Request):
    email = require_email(request)
    try:
        store.delete_chat(email, chat_id)
    except Exception as e:
        _log_error("delete_chat", e)
        raise HTTPException(502, "Couldn't delete this chat — Supabase is unreachable.")
    return {"ok": True}


class PinBody(BaseModel):
    pinned: bool


@app.put("/api/chats/{chat_id}/pin")
def set_pinned(chat_id: str, body: PinBody, request: Request):
    email = require_email(request)
    try:
        if body.pinned:
            if store.count_pinned(email) >= store.MAX_PINNED_CHATS:
                raise HTTPException(409, f"You can only pin up to {store.MAX_PINNED_CHATS} chats.")
            store.pin_chat(email, chat_id)
        else:
            store.unpin_chat(email, chat_id)
    except HTTPException:
        raise
    except Exception as e:
        _log_error("set_pinned", e)
        raise HTTPException(502, "Couldn't update this chat — Supabase is unreachable.")
    return {"ok": True}


# --- Chat (streaming) ---

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatBody(BaseModel):
    chat_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    # The client sends the conversation so far, so anonymous (unsaved) chats
    # work the same way as saved ones.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)


@app.post("/api/chat")
def chat(body: ChatBody, request: Request):
    """Streams newline-delimited JSON events:

        {"type": "sources", "sources": [...]}   once, before the answer
        {"type": "token", "text": "..."}        many
        {"type": "title", "title": "..."}       first message of a saved chat only
        {"type": "error", "message": "..."}     if generation fails
        {"type": "done"}
    """
    user = current_user(request)
    email = user["email"] if user else None
    history = [("human" if m.role == "user" else "ai", m.content) for m in body.history]

    title_future = None
    if email:
        try:
            store.save_message(email, body.chat_id, "user", body.message)
        except Exception as e:
            _log_error("save_message:user", e)
        if not body.history:
            # Runs alongside the answer instead of delaying the first token.
            title_future = _title_pool.submit(rag.generate_chat_title, body.message)

    def events():
        answer_parts: list[str] = []
        try:
            for event in rag.stream_answer(body.message, history):
                if event["type"] == "token":
                    answer_parts.append(event["text"])
                yield json.dumps(event) + "\n"
        except Exception as e:
            _log_error("stream_answer", e)
            yield json.dumps({
                "type": "error",
                "message": "Something went wrong generating a response. Please try sending your question again.",
            }) + "\n"
            return

        answer = "".join(answer_parts) or "No response generated."
        if email:
            try:
                store.save_message(email, body.chat_id, "assistant", answer)
            except Exception as e:
                _log_error("save_message:assistant", e)
            if title_future:
                title = title_future.result()
                try:
                    store.save_chat_title(email, body.chat_id, title)
                except Exception as e:
                    _log_error("save_chat_title", e)
                yield json.dumps({"type": "title", "title": title}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


# --- Frontend ---
# The built React app (npm --prefix frontend run build). Low priority, so every
# /api route above wins; unknown paths like /c/<id> fall back to index.html so
# client-side routing works on refresh. On Vercel these files are promoted to
# the CDN at build time. In local dev Vite serves the frontend instead, so the
# directory may not exist.
app.frontend("/", directory="frontend/dist", check_dir=False)
