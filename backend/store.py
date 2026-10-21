"""Per-user chat persistence in Supabase (server-side only, service_role key)."""
from functools import lru_cache

from supabase import create_client, Client

from .config import SUPABASE_URL, SUPABASE_SERVICE_KEY

MAX_PINNED_CHATS = 5


@lru_cache
def db() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def list_chats(email: str) -> list[dict]:
    """Every chat for a user, newest first, with title and pinned flag."""
    rows = (
        db().table("chat_messages")
        .select("chat_id, role, content, created_at")
        .eq("user_email", email)
        .order("created_at")
        .execute()
    ).data or []

    chats: dict[str, dict] = {}
    for row in rows:
        chat = chats.setdefault(row["chat_id"], {
            "id": row["chat_id"],
            "title": None,
            "started_at": row["created_at"],
            "pinned_at": None,
        })
        if chat["title"] is None and row["role"] == "user":
            content = row["content"]
            chat["title"] = content[:40] + ("…" if len(content) > 40 else "")

    titles = db().table("chat_titles").select("chat_id, title").eq("user_email", email).execute().data or []
    for row in titles:
        if row["chat_id"] in chats:
            chats[row["chat_id"]]["title"] = row["title"]

    pins = db().table("pinned_chats").select("chat_id, pinned_at").eq("user_email", email).execute().data or []
    for row in pins:
        if row["chat_id"] in chats:
            chats[row["chat_id"]]["pinned_at"] = row["pinned_at"]

    return sorted(chats.values(), key=lambda c: c["started_at"], reverse=True)


def get_messages(email: str, chat_id: str) -> list[dict]:
    rows = (
        db().table("chat_messages")
        .select("role, content")
        .eq("user_email", email)
        .eq("chat_id", chat_id)
        .order("id")
        .execute()
    ).data or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def save_message(email: str, chat_id: str, role: str, content: str):
    db().table("chat_messages").insert(
        {"user_email": email, "chat_id": chat_id, "role": role, "content": content}
    ).execute()


def save_chat_title(email: str, chat_id: str, title: str):
    db().table("chat_titles").upsert({"chat_id": chat_id, "user_email": email, "title": title}).execute()


def delete_chat(email: str, chat_id: str):
    for table in ("chat_messages", "pinned_chats", "chat_titles"):
        db().table(table).delete().eq("user_email", email).eq("chat_id", chat_id).execute()


def count_pinned(email: str) -> int:
    rows = db().table("pinned_chats").select("chat_id").eq("user_email", email).execute().data or []
    return len(rows)


def pin_chat(email: str, chat_id: str):
    db().table("pinned_chats").upsert(
        {"user_email": email, "chat_id": chat_id}, on_conflict="user_email,chat_id"
    ).execute()


def unpin_chat(email: str, chat_id: str):
    db().table("pinned_chats").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
