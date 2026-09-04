import streamlit as st
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import json
from supabase import create_client
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1. SETUP & LOADERS
load_dotenv()


@st.cache_data
def load_mappings():
    try:
        with open("youtube_links.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


url_map = load_mappings()


def _log_error(context: str, exc: Exception):
    with open("app_errors.log", "a") as f:
        f.write(f"[{context}] {exc!r}\n")


# 1b. PER-USER CHAT PERSISTENCE (Supabase)
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["service_key"])


@st.cache_data(ttl=60)
def load_all_histories(email: str) -> dict[str, list[dict]]:
    """All of a user's messages in one query, grouped by chat_id — reused
    for both the sidebar (via list_chats) and opening any individual chat,
    so switching between chats doesn't cost a fresh round-trip per click.
    """
    try:
        res = (
            get_supabase()
            .table("chat_messages")
            .select("chat_id, role, content")
            .eq("user_email", email)
            .order("id")
            .execute()
        )
    except Exception as e:
        _log_error("load_all_histories", e)
        st.toast("Couldn't load saved chats — Supabase is unreachable.", icon=":material/error:")
        return {}
    histories: dict[str, list[dict]] = {}
    for row in res.data or []:
        histories.setdefault(row["chat_id"], []).append(
            {"role": row["role"], "content": row["content"]}
        )
    return histories


def save_message(email: str, chat_id: str, role: str, content: str):
    try:
        get_supabase().table("chat_messages").insert(
            {"user_email": email, "chat_id": chat_id, "role": role, "content": content}
        ).execute()
    except Exception as e:
        _log_error("save_message", e)
        st.toast("This message wasn't saved — Supabase is unreachable.", icon=":material/error:")
        return
    load_all_histories.clear()
    list_chats.clear()


def delete_chat(email: str, chat_id: str):
    try:
        get_supabase().table("chat_messages").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
        get_supabase().table("pinned_chats").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
        get_supabase().table("chat_titles").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
        get_supabase().table("message_feedback").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
    except Exception as e:
        _log_error("delete_chat", e)
        st.toast("Couldn't delete this chat — Supabase is unreachable.", icon=":material/error:")
        return
    list_chats.clear()
    list_pinned_chat_ids.clear()
    load_all_histories.clear()
    load_feedback.clear()


MAX_PINNED_CHATS = 5


@st.cache_data(ttl=60)
def list_pinned_chat_ids(email: str) -> list[str]:
    try:
        res = (
            get_supabase()
            .table("pinned_chats")
            .select("chat_id")
            .eq("user_email", email)
            .order("pinned_at", desc=True)
            .execute()
        )
    except Exception as e:
        _log_error("list_pinned_chat_ids", e)
        return []
    return [row["chat_id"] for row in (res.data or [])]


def pin_chat(email: str, chat_id: str):
    try:
        get_supabase().table("pinned_chats").insert(
            {"user_email": email, "chat_id": chat_id}
        ).execute()
    except Exception as e:
        _log_error("pin_chat", e)
        st.toast("Couldn't pin this chat — Supabase is unreachable.", icon=":material/error:")
        return
    list_pinned_chat_ids.clear()


def unpin_chat(email: str, chat_id: str):
    try:
        get_supabase().table("pinned_chats").delete().eq("user_email", email).eq("chat_id", chat_id).execute()
    except Exception as e:
        _log_error("unpin_chat", e)
        st.toast("Couldn't unpin this chat — Supabase is unreachable.", icon=":material/error:")
        return
    list_pinned_chat_ids.clear()


@st.cache_data(ttl=60)
def list_chats(email: str) -> list[dict]:
    try:
        res = (
            get_supabase()
            .table("chat_messages")
            .select("chat_id, role, content, created_at")
            .eq("user_email", email)
            .order("created_at")
            .execute()
        )
    except Exception as e:
        _log_error("list_chats", e)
        st.toast("Couldn't load your chat list — Supabase is unreachable.", icon=":material/error:")
        return []
    chats: dict[str, dict] = {}
    for row in res.data or []:
        chat = chats.setdefault(row["chat_id"], {
            "chat_id": row["chat_id"],
            "title": None,
            "started_at": row["created_at"],
        })
        if chat["title"] is None and row["role"] == "user":
            content = row["content"]
            chat["title"] = content[:40] + ("…" if len(content) > 40 else "")

    try:
        titles_res = (
            get_supabase()
            .table("chat_titles")
            .select("chat_id, title")
            .eq("user_email", email)
            .execute()
        )
        for row in titles_res.data or []:
            if row["chat_id"] in chats:
                chats[row["chat_id"]]["title"] = row["title"]
    except Exception as e:
        _log_error("list_chats:titles", e)

    return sorted(chats.values(), key=lambda c: c["started_at"], reverse=True)


def save_chat_title(email: str, chat_id: str, title: str):
    try:
        get_supabase().table("chat_titles").upsert(
            {"chat_id": chat_id, "user_email": email, "title": title}
        ).execute()
    except Exception as e:
        _log_error("save_chat_title", e)
        return
    list_chats.clear()


@st.cache_data(ttl=60)
def load_feedback(chat_id: str) -> dict[int, str]:
    """message_index -> 'up'/'down' for every rated assistant message in a chat."""
    try:
        res = (
            get_supabase()
            .table("message_feedback")
            .select("message_index, rating")
            .eq("chat_id", chat_id)
            .execute()
        )
    except Exception as e:
        _log_error("load_feedback", e)
        return {}
    return {row["message_index"]: row["rating"] for row in (res.data or [])}


def save_feedback(email: str, chat_id: str, message_index: int, question: str, answer: str, rating: str):
    try:
        get_supabase().table("message_feedback").upsert(
            {
                "user_email": email,
                "chat_id": chat_id,
                "message_index": message_index,
                "question": question,
                "answer": answer,
                "rating": rating,
            },
            on_conflict="chat_id,message_index",
        ).execute()
    except Exception as e:
        _log_error("save_feedback", e)
        st.toast("Couldn't save your feedback — Supabase is unreachable.", icon=":material/error:")
        return
    load_feedback.clear()


@st.cache_resource
def get_title_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3, max_output_tokens=30)


def generate_chat_title(question: str) -> str:
    fallback = question[:40] + ("…" if len(question) > 40 else "")
    try:
        prompt = (
            "You're titling conversations in a vertical jump / dunk training chatbot, so every "
            "message is already about that topic — don't restate it. Reply with exactly ONE "
            "short chat title (3-6 words) capturing what's specifically being asked in this "
            "message. Output only that single title — no alternatives, no bullets, no quotes, "
            "no trailing punctuation:\n\n"
            f"{question}"
        )
        raw_content = get_title_llm().invoke(prompt).content
        if isinstance(raw_content, list):
            raw_content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in raw_content
            )
        # Guard against the model still returning multiple lines/bullets despite
        # the "exactly ONE title" instruction — take just the first non-empty one.
        title = next((line.strip() for line in raw_content.splitlines() if line.strip()), "")
        title = title.lstrip("*-•").strip().strip('"').strip()
        return title[:60] if title else fallback
    except Exception as e:
        with open("title_gen_errors.log", "a") as f:
            f.write(f"{e!r}\n")
        return fallback


# 2. RAG ENGINE (Importable Logic)

# Only THP Strength and John Evans are actual channels/coaches — their videos
# are tagged with a "coach" metadata field ("THP" or "John") that retrieval
# can filter on. Guests below aren't tied to one channel, so they're handled
# via the [SPEAKER: ...] tags in the text instead, not a retrieval filter.
KNOWN_GUESTS = [
    "Josh Ruble",
    "Dom Gonzales (aka Dom Dunks)",
    "Donovan Hawkins",
    "Austin",
    "Ben Moxness",
]


def infer_coach_filter(question: str) -> str | None:
    """Best-effort guess at which channel's videos to constrain retrieval to."""
    q = question.lower()
    if "evans" in q:
        return "John"
    if "thp" in q:
        return "THP"
    return None


def apply_coach_filter(retriever, question: str):
    coach = infer_coach_filter(question)
    if coach:
        retriever.search_kwargs["filter"] = {"coach": coach}
    else:
        retriever.search_kwargs.pop("filter", None)


@st.cache_resource
def init_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_query",
    )
    vectorstore = Pinecone(index_name="vetted-vertical", embedding=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rephrase the "
         "question into a standalone question that can be understood without "
         "the chat history. Do not answer it, just reformulate it if needed "
         "and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    system_prompt = (
        "You are an Elite Vertical Jump Coach specializing in THP Strength and John Evans methodologies. "
        "Analyze the video transcripts provided below and give a technical answer.\n\n"
        "Some clips feature a guest or athlete speaking in first person about their own training or "
        "results, not the channel's coach — recurring guests include: " + ", ".join(KNOWN_GUESTS) + ". "
        "Each chunk of context below is prefixed with its video title and coach/channel — use that to "
        "tell apart content from different videos or sources. The text may also contain inline tags "
        "like '[SPEAKER: <name>]' marking who is talking in the text that follows. Only attribute a "
        "specific person's stats, results, or experiences to them if a [SPEAKER: ...] tag or explicit "
        "statement in the text confirms it — match names/aliases above to the same person. If the "
        "relevant text is tagged [SPEAKER: Uncertain], has no speaker tag, or you otherwise can't tell "
        "whose stats or experience is being described, say so instead of guessing which coach or "
        "athlete it refers to.\n\n"
        "FORMATTING: Never bold the lead-in phrase of a bullet point (e.g. do NOT write "
        "'* **Tendon response:** the tendon adapts...' — write '* The tendon adapts...' instead). "
        "Bullets should be plain text. Reserve **bold** only for a single specific number or term "
        "elsewhere in the answer that's genuinely easy to miss, and only if truly necessary — most "
        "answers should have zero bolded text.\n\n"
        "VIDEO CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    document_prompt = PromptTemplate.from_template(
        "[Video: {video_title} | Coach: {coach} | Focus: {primary_focus} | "
        "Difficulty: {difficulty} | Exercises: {exercise_list} | Stats: {stats_summary}]\n"
        "{page_content}"
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    return chain, retriever


# This function is what auditor.py will use
def query_rag(user_input: str, chat_history: list | None = None):
    chain, retriever = init_rag_chain()
    apply_coach_filter(retriever, user_input)
    response = chain.invoke({"input": user_input, "chat_history": chat_history or []})
    return {
        "answer": response.get("answer"),
        "context": [doc.page_content for doc in response.get("context", [])]
    }


# 3. UI CODE (Browser Only)
if __name__ == "__main__":
    st.set_page_config(page_title="Dunk Bot", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            max-width: 960px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 3rem;
        }
        [data-testid="stChatMessage"] {
            width: fit-content;
            max-width: 100%;
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
            background: transparent;
            border: none;
            box-shadow: none;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            font-weight: 300 !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"],
        [data-testid="stChatMessageAvatarCustom"] {
            display: none;
        }
        div[class*="st-key-row-user-"] [data-testid="stChatMessage"] {
            margin-left: auto;
            background: #282828;
            border-radius: 18px;
            padding: 0.65rem 1.1rem;
        }
        div[class*="st-key-row-assistant-"] [data-testid="stChatMessage"] {
            margin-right: auto;
            max-width: 700px;
        }
        div[class*="st-key-row-assistant-"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.65;
        }
        div[class*="st-key-row-assistant-"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.85em;
        }
        div[class*="st-key-row-assistant-"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0;
        }
        [data-testid="stChatInput"] {
            border-radius: 28px;
            max-width: 820px;
            margin-left: auto;
            margin-right: auto;
        }
        [data-testid="stChatInput"] [class*="e15xmbo01"] {
            padding-top: 6px !important;
            padding-bottom: 6px !important;
            border-width: 1px !important;
            border-color: #454545 !important;
            background-color: #1E1E1E !important;
            border-radius: 28px !important;
            box-shadow: none !important;
        }
        /* Default is a solid filled primary-color (red) circle -- restyle to
           a quiet, borderless icon that blends into the input, matching a
           minimal composer-bar look instead of a bright CTA button. */
        [data-testid="stChatInputSubmitButton"] {
            background-color: transparent !important;
            color: #8E8E8E !important;
        }
        [data-testid="stChatInputSubmitButton"]:not(:disabled):hover {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #B0B0B0 !important;
        }
        [data-testid="stChatInputSubmitButton"]:disabled {
            color: #4A4A4A !important;
        }
        [data-testid="stChatInput"] div:has(> [data-testid="stChatInputTextArea"]),
        [data-testid="stChatInput"] div:has(> div > [data-testid="stChatInputTextArea"]) {
            background-color: transparent !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: #8A8A8A !important;
            font-weight: 300 !important;
        }
        div[class*="st-key-chatrow-"] [data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
            gap: 0 !important;
        }
        div[class*="st-key-chatrow-"] [data-testid="stColumn"] {
            padding: 0 !important;
        }
        /* Force every wrapper between chatrow and the title button to
           position:static, so the button's position:absolute below is
           guaranteed to anchor to chatrow itself (which is what actually
           gets the hover/selected highlight) — regardless of whatever
           position Streamlit's own internal classes set on these. */
        div[class*="st-key-chatrow-"] [data-testid="stLayoutWrapper"],
        div[class*="st-key-chatrow-"] [data-testid="stHorizontalBlock"],
        div[class*="st-key-chatrow-"] [data-testid="stColumn"],
        div[class*="st-key-chatrow-"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        div[class*="st-key-chat-"],
        div[class*="st-key-chat-"] .stButton {
            position: static !important;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            z-index: 1;
            border-radius: 8px;
            padding: 0px 8px !important;
            min-height: 0 !important;
            transition: background-color 0.15s ease;
            position: absolute !important;
            inset: 0 !important;
            display: flex !important;
            align-items: center !important;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] p {
            text-align: left;
            color: #B0B0B0;
            line-height: 1;
            margin: 0 !important;
            font-size: 0.9rem;
            font-weight: 300 !important;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] p::before {
            content: "";
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            border: 1px solid #666666;
            margin-right: 8px;
            vertical-align: middle;
            flex-shrink: 0;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"]:not(:disabled):hover {
            background: rgba(255, 255, 255, 0.08) !important;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"]:disabled {
            background: transparent !important;
            opacity: 1 !important;
            cursor: default;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"]:disabled p {
            color: #F2F2F2 !important;
        }
        .st-key-new-chat-btn button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 8px;
            transition: background-color 0.15s ease;
        }
        .st-key-new-chat-btn button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start;
        }
        .st-key-new-chat-btn button[data-testid="stBaseButton-secondary"]:hover {
            background: rgba(255, 255, 255, 0.08) !important;
        }
        .st-key-new-chat-btn [data-testid="stIconMaterial"] {
            font-size: 13px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 20px !important;
            height: 20px !important;
            border-radius: 50% !important;
            background: rgba(255, 255, 255, 0.12) !important;
            position: relative !important;
            top: 1px !important;
            flex-shrink: 0 !important;
        }
        .st-key-see-more-chats button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 2px 4px !important;
            min-height: 0 !important;
            opacity: 0.5;
            transition: opacity 0.15s ease;
        }
        .st-key-see-more-chats button[data-testid="stBaseButton-secondary"]:hover {
            opacity: 0.85 !important;
        }
        .st-key-see-more-chats button[data-testid="stBaseButton-secondary"] p {
            font-size: 0.78rem;
        }
        [data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        [data-testid="stSidebarUserContent"] {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            padding-top: 12px !important;
            padding-bottom: 0 !important;
        }
        [data-testid="stSidebarUserContent"] > div {
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }
        [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
            display: flex !important;
            flex-direction: column;
            flex: 1;
            min-height: 0;
        }
        [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar-settings) {
            margin-top: auto;
        }
        .st-key-sidebar-settings {
            padding-bottom: 3px !important;
            margin-left: -14.5px;
            width: calc(100% + 14.5px);
        }
        /* Strip the default button box (background/border) so the account
           row reads as plain content with a hover highlight, not a button,
           and add the letter-avatar circle as a pseudo-element since
           st.popover's label can't hold raw HTML. */
        .st-key-sidebar-settings [data-testid="stPopover"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 8px;
            padding: 6px 8px 6px 4px !important;
            justify-content: flex-start !important;
            transition: background-color 0.15s ease;
        }
        .st-key-sidebar-settings [data-testid="stPopover"] button:hover {
            background: rgba(255, 255, 255, 0.08) !important;
        }
        .st-key-sidebar-settings [data-testid="stPopover"] button::before {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #3A3A3A;
            color: #F2F2F2;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
            flex-shrink: 0;
        }
        [data-testid="stSidebar"] {
            background-color: #121212 !important;
            border-right: 0.5px solid #333333;
            min-width: 260px !important;
            position: relative;
        }
        /* The collapse arrow lives in its own header container, sitting in
           a separate row above "New" -- collapse that header out of the
           layout entirely and float the arrow itself into "New"'s row on
           the right instead. */
        [data-testid="stLogoSpacer"] {
            display: none !important;
        }
        [data-testid="stSidebarHeader"] {
            position: absolute !important;
            top: 0;
            left: 0;
            width: 100% !important;
            height: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            z-index: 3;
        }
        [data-testid="stSidebarCollapseButton"] {
            position: absolute !important;
            top: 17.25px;
            right: 8px;
        }
        [data-testid="stSidebarCollapseButton"] button {
            visibility: visible !important;
        }
        [data-testid="stSidebar"] button p {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* st.divider() renders an <hr> whose own browser-default margin plus
           Streamlit's element-container spacing add up to a lot of empty
           space around a single line -- collapse both down. */
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(hr) {
            margin: 0 !important;
        }
        [data-testid="stSidebar"] hr {
            margin: 8px 0 !important;
        }
        .st-key-previous-chats-list,
        .st-key-pinned-chats-list {
            gap: 0 !important;
        }
        .st-key-previous-chats-list [data-testid="stElementContainer"],
        .st-key-pinned-chats-list [data-testid="stElementContainer"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-pinned-section,
        .st-key-chats-section {
            gap: 0 !important;
        }
        .st-key-pinned-section [data-testid="stElementContainer"],
        .st-key-chats-section [data-testid="stElementContainer"] {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 4px 0 !important;
            min-height: 0 !important;
            justify-content: flex-start !important;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start !important;
            align-items: center;
            width: fit-content;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"] p {
            font-size: 0.8rem;
            font-weight: 340;
            color: #8A8A8A;
        }
        div[class*="st-key-section-toggle-"] [data-testid="stIconMaterial"] {
            display: none !important;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"] > div::after {
            content: "▸";
            display: inline-block;
            margin-left: 4px;
            font-size: 16px;
            line-height: 1;
            color: #8A8A8A;
            opacity: 0;
            position: relative;
            top: -1px;
            left: 6px;
            transition: opacity 0.15s ease;
        }
        div[class*="st-key-section-toggle-"][class*="-open"] button[data-testid="stBaseButton-secondary"] > div::after {
            content: "▾";
            top: -1px;
            left: 6px;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"]:hover > div::after {
            opacity: 1;
        }
        div[class*="st-key-section-toggle-"] button[data-testid="stBaseButton-secondary"]:hover p {
            color: #B0B0B0 !important;
        }
        /* Zeroes the .stButton wrapper div itself (a real Streamlit class,
           not just the testid) — it carries its own default padding that
           was adding ~3.5px above/below the button regardless of the
           button's own padding/height being zeroed. */
        div[class*="st-key-menu-"] {
            padding: 0 !important;
            margin: 0 !important;
            line-height: 0 !important;
        }
        div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            min-height: 0 !important;
            height: 15px !important;
            line-height: 1 !important;
            padding: 0 !important;
            opacity: 0 !important;
            transition: opacity 0.15s ease, background-color 0.15s ease;
            position: relative;
            top: 7px;
        }
        div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"]:hover {
            background: rgba(255, 255, 255, 0.15) !important;
        }
        div[class*="st-key-menu-"] [data-testid="stIconMaterial"] {
            font-size: 14px !important;
            line-height: 1 !important;
        }
        div[class*="st-key-chatrow-"] {
            border-radius: 8px;
            transition: background-color 0.15s ease;
            position: relative;
        }
        div[class*="st-key-menuwrap-"] {
            position: relative;
            z-index: 2;
        }
        div[class*="st-key-menucontent-"] {
            display: none;
            position: absolute;
            top: 100%;
            right: 0;
            margin-top: 4px;
            background: #1E1E1E;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 4px;
            width: fit-content !important;
            min-width: 140px;
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        }
        div[class*="st-key-menucontent-"] button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start !important;
        }
        div[class*="st-key-menuwrap-"]:focus-within div[class*="st-key-menucontent-"] {
            display: block !important;
        }
        div[class*="st-key-chatrow-"]:hover,
        div[class*="st-key-chatrow-"]:has(button:disabled) {
            background: rgba(255, 255, 255, 0.08) !important;
        }
        div[class*="st-key-chatrow-"]:hover div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"],
        div[class*="st-key-menuwrap-"]:focus-within div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"] {
            opacity: 1 !important;
        }
        div[class*="st-key-chatrow-"] div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"]:not(:disabled):hover {
            background: transparent !important;
        }
        .st-key-new-chat-hero [data-testid="stChatInput"] {
            max-width: 600px;
        }
        /* The spinner text sits inside a wrapper whose own box height is
           unstable (collapses near 0 and the text overflows it) -- fighting
           that with align-items/line-height had zero measured effect across
           repeated tests. What *is* reliable: the icon consistently renders
           exactly 7px above the text's vertical center, measured directly
           via getBoundingClientRect across many trials. Nudging the icon
           down by that fixed, empirically-verified amount is the actual fix. */
        [data-testid="stSpinner"] [data-testid="stMarkdownContainer"] p {
            line-height: 1 !important;
            margin: 0 !important;
        }
        [data-testid="stSpinner"] [data-testid="stSpinnerIcon"] {
            position: relative;
            top: 7px;
        }
        div[class*="st-key-fb-up-"] button[data-testid="stBaseButton-secondary"],
        div[class*="st-key-fb-down-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 2px 6px !important;
            min-height: 0 !important;
            opacity: 0.55;
            transition: opacity 0.15s ease;
        }
        div[class*="st-key-fb-up-"] button[data-testid="stBaseButton-secondary"]:hover,
        div[class*="st-key-fb-down-"] button[data-testid="stBaseButton-secondary"]:hover {
            opacity: 1;
        }
        div[class*="st-key-fb-up-"] button[data-testid="stBaseButton-secondary"]:disabled,
        div[class*="st-key-fb-down-"] button[data-testid="stBaseButton-secondary"]:disabled {
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @contextmanager
    def chat_bubble(role: str, key_suffix):
        with st.container(key=f"row-{role}-{key_suffix}"):
            with st.chat_message(role):
                yield

    def render_feedback_buttons(user_email, chat_id, message_index, question, answer):
        current = load_feedback(chat_id).get(message_index)
        up_col, down_col, _ = st.columns([1, 1, 10])
        with up_col:
            if st.button("👍", key=f"fb-up-{chat_id}-{message_index}", disabled=current == "up"):
                save_feedback(user_email, chat_id, message_index, question, answer, "up")
                st.rerun()
        with down_col:
            if st.button("👎", key=f"fb-down-{chat_id}-{message_index}", disabled=current == "down"):
                save_feedback(user_email, chat_id, message_index, question, answer, "down")
                st.rerun()

    def render_chat_row(chat, user_email, is_pinned):
        chat_id = chat["chat_id"]
        is_current = chat_id == st.session_state.chat_id
        label = chat["title"] or "(empty chat)"
        with st.container(key=f"chatrow-{chat_id}"):
            title_col, menu_col = st.columns([6, 1], vertical_alignment="center")
            with title_col:
                if st.button(label, key=f"chat-{chat_id}", use_container_width=True,
                             disabled=is_current):
                    st.session_state.chat_id = chat_id
                    st.session_state.messages = None  # sentinel: not loaded yet
                    st.session_state.view = "chat"
                    st.rerun()
            with menu_col:
                # Menu visibility is handled purely by CSS (:focus-within on
                # menuwrap below) — clicking elsewhere blurs the trigger and
                # closes it natively, no session_state toggle needed.
                with st.container(key=f"menuwrap-{chat_id}"):
                    st.button("", key=f"menu-{chat_id}", icon=":material/more_vert:")
                    with st.container(key=f"menucontent-{chat_id}"):
                        if is_pinned:
                            if st.button("Unpin", key=f"unpin-{chat_id}", icon=":material/keep_off:",
                                         use_container_width=True):
                                unpin_chat(user_email, chat_id)
                                st.rerun()
                        else:
                            if st.button("Pin", key=f"pin-{chat_id}", icon=":material/push_pin:",
                                         use_container_width=True):
                                if len(list_pinned_chat_ids(user_email)) >= MAX_PINNED_CHATS:
                                    st.toast(f"You can only pin up to {MAX_PINNED_CHATS} chats.",
                                             icon=":material/error:")
                                else:
                                    pin_chat(user_email, chat_id)
                                    st.rerun()
                        if st.button("Delete", key=f"delete-{chat_id}", icon=":material/delete:",
                                     use_container_width=True):
                            delete_chat(user_email, chat_id)
                            if is_current:
                                st.session_state.chat_id = str(uuid.uuid4())
                                st.session_state.messages = []
                            st.rerun()

    # --- AUTH GATE ---
    try:
        auth_configured = "auth" in st.secrets
    except StreamlitSecretNotFoundError:
        auth_configured = False

    if not auth_configured:
        st.error(
            "Google login isn't configured yet. Copy "
            "`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
            "and fill in your own values."
        )
        st.stop()

    # Logging in is optional — anonymous users get a fully working chat that
    # just isn't saved anywhere (no Supabase history, pinning, titles, or
    # feedback persistence). user_email is None for the whole rest of the
    # app in that case, and every persistence call below is guarded on it.
    user_email = st.user.email if st.user.is_logged_in else None
    first_name = (
        (st.user.get("given_name") or st.user.get("name") or "Your") if user_email else "Your"
    )

    # A fresh chat_id each time the session starts (e.g. on login) — never
    # auto-resumes a previous conversation.
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []

    # A chat was just clicked into (see render_chat_row / all-chats search):
    # fetch its messages *before* reconstructing the sidebar, so a chat that's
    # already cached (the common case) resolves in this same pass with no
    # extra round-trip — only a genuinely slow, uncached fetch shows a spinner.
    if st.session_state.messages is None:
        if user_email:
            with st.spinner("Loading chat..."):
                st.session_state.messages = load_all_histories(user_email).get(st.session_state.chat_id, [])
        else:
            st.session_state.messages = []

    try:
        rag_chain, retriever = init_rag_chain()
    except Exception as e:
        _log_error("init_rag_chain", e)
        st.error(
            "Couldn't connect to the retrieval backend (Gemini/Pinecone). "
            "Please try again shortly."
        )
        st.stop()

    # "chat" = normal chat UI, "all_chats" = the full searchable chat list.
    if "view" not in st.session_state:
        st.session_state.view = "chat"

    # --- SIDEBAR & DEBUGGER ---
    with st.sidebar:
        if st.button("New", key="new-chat-btn", icon=":material/add:", use_container_width=True):
            st.session_state.chat_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.view = "chat"
            st.rerun()

        st.divider()

        if user_email:
            all_chats = list_chats(user_email)
            chats_by_id = {chat["chat_id"]: chat for chat in all_chats}
            pinned_ids = [cid for cid in list_pinned_chat_ids(user_email) if cid in chats_by_id]

            if "pinned_section_open" not in st.session_state:
                st.session_state.pinned_section_open = True
            if "chats_section_open" not in st.session_state:
                st.session_state.chats_section_open = True

            if pinned_ids:
                with st.container(key="pinned-section"):
                    state = "open" if st.session_state.pinned_section_open else "closed"
                    with st.container(key=f"section-toggle-pinned-{state}"):
                        if st.button("Pinned", key="toggle-pinned-section", use_container_width=True):
                            st.session_state.pinned_section_open = not st.session_state.pinned_section_open
                            st.rerun()
                    if st.session_state.pinned_section_open:
                        with st.container(key="pinned-chats-list"):
                            for chat_id in pinned_ids:
                                render_chat_row(chats_by_id[chat_id], user_email, is_pinned=True)

            with st.container(key="chats-section"):
                state = "open" if st.session_state.chats_section_open else "closed"
                with st.container(key=f"section-toggle-chats-{state}"):
                    if st.button(f"{first_name}'s Chats", key="toggle-chats-section", use_container_width=True):
                        st.session_state.chats_section_open = not st.session_state.chats_section_open
                        st.rerun()
                if st.session_state.chats_section_open:
                    with st.container(key="previous-chats-list"):
                        unpinned_chats = [c for c in all_chats if c["chat_id"] not in pinned_ids]
                        for chat in unpinned_chats[:12]:
                            render_chat_row(chat, user_email, is_pinned=False)

                        if len(unpinned_chats) > 12:
                            if st.button("View all conversations", key="see-more-chats"):
                                st.session_state.view = "all_chats"
                                st.rerun()
        else:
            st.caption("Log in to save your chat history, pin chats, and leave feedback on answers.")

        with st.container(key="sidebar-settings"):
            st.divider()
            if user_email:
                # st.popover can't take raw HTML in its label, so the letter
                # avatar is injected as a ::before pseudo-element instead --
                # this tiny stylesheet is the only place that needs the
                # actual initial, computed fresh per user.
                initial = (first_name[0] if first_name and first_name != "Your" else user_email[0]).upper()
                st.markdown(
                    f"<style>.st-key-sidebar-settings [data-testid=\"stPopover\"] "
                    f"button::before {{ content: \"{initial}\"; }}</style>",
                    unsafe_allow_html=True,
                )
                with st.popover(first_name, use_container_width=True):
                    if st.button("Log out", key="logout-btn", icon=":material/logout:", use_container_width=True):
                        st.logout()
            else:
                if st.button("Log in with Google", icon=":material/login:"):
                    st.login()

    # --- ALL CHATS (SEARCH) PAGE ---
    if st.session_state.view == "all_chats" and user_email:
        if st.button("← Back", key="back-to-chat"):
            st.session_state.view = "chat"
            st.rerun()

        st.title(f"All of {first_name}'s Chats")
        search_query = st.text_input("Search chats", placeholder="Search by title...")

        all_chats = list_chats(user_email)
        if search_query:
            all_chats = [
                c for c in all_chats
                if search_query.lower() in (c["title"] or "").lower()
            ]

        if not all_chats:
            st.caption("No chats found.")

        for chat in all_chats:
            label = chat["title"] or "(empty chat)"
            if st.button(label, key=f"allchat-{chat['chat_id']}", use_container_width=True):
                st.session_state.chat_id = chat["chat_id"]
                st.session_state.messages = None  # sentinel: not loaded yet
                st.session_state.view = "chat"
                st.rerun()

    # --- CHAT DISPLAY ---
    else:
        is_new_chat = not st.session_state.messages
        hero_placeholder = st.empty()

        if is_new_chat:
            with hero_placeholder.container():
                with st.container(key="new-chat-hero"):
                    # Reserve the title's slot first (so it stays visually above
                    # the input), but defer writing to it until we know whether
                    # this run is actually a submission.
                    title_slot = st.empty()
                    hero_input = st.chat_input("Ask a technical training question...", key="chat_input_hero")
                    if not hero_input:
                        title_slot.markdown(
                            "<h1 style='text-align: center; margin-top: 30vh; margin-bottom: 1.5rem; "
                            "font-size: 2.6rem; font-weight: 500;'>Ask me anything about dunking</h1>",
                            unsafe_allow_html=True,
                        )

            if hero_input:
                # Clear the hero right now, in this same run, instead of relying
                # on a rerun to prune it — that was slow and left stale content
                # on screen for the whole RAG call. We still rerun immediately
                # after saving the message (before generating a response) so
                # the sidebar picks up this new chat right away, by send time,
                # rather than only after the bot's reply finishes.
                hero_placeholder.empty()
                st.session_state.messages.append({"role": "user", "content": hero_input})
                if user_email:
                    save_message(user_email, st.session_state.chat_id, "user", hero_input)
                    title = generate_chat_title(hero_input)
                    save_chat_title(user_email, st.session_state.chat_id, title)
                st.rerun()

        if not is_new_chat:
            for i, message in enumerate(st.session_state.messages):
                with chat_bubble(message["role"], i):
                    st.markdown(message["content"])
                if message["role"] == "assistant":
                    # Sources are only ever available for the message just
                    # generated this session (never persisted/reloaded from
                    # Supabase) — stashed by the generation step below and
                    # shown here, once, the first time this becomes the
                    # newest message.
                    if i == len(st.session_state.messages) - 1 and "pending_sources" in st.session_state:
                        with st.expander("Sources & References"):
                            for doc in st.session_state.pending_sources:
                                title = doc.metadata.get("video_title", "Untitled Video")
                                video_id = doc.metadata.get("video_id")
                                start_time = doc.metadata.get("start_time")

                                if video_id and start_time is not None:
                                    url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                                else:
                                    url = url_map.get(video_id, "https://www.youtube.com/@thpstrength1/videos")

                                st.markdown(f"🔗 [{title}]({url})")
                        del st.session_state["pending_sources"]

                    if user_email:
                        preceding_question = (
                            st.session_state.messages[i - 1]["content"] if i > 0 else ""
                        )
                        render_feedback_buttons(
                            user_email, st.session_state.chat_id, i, preceding_question, message["content"]
                        )

            user_input = st.chat_input("Ask a technical training question...", key="chat_input_bottom")

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                if user_email:
                    save_message(user_email, st.session_state.chat_id, "user", user_input)
                with chat_bubble("user", "new"):
                    st.markdown(user_input)

            # --- RESPOND TO THE MOST RECENT UNANSWERED USER MESSAGE ---
            # Covers a fresh send above, and a message just carried over from
            # the new-chat hero (already appended, not yet answered).
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                pending_question = st.session_state.messages[-1]["content"]
                chat_history = [
                    ("human" if m["role"] == "user" else "ai", m["content"])
                    for m in st.session_state.messages[:-1]
                ]

                with chat_bubble("assistant", "new"):
                    with st.spinner("Retrieving video data..."):
                        try:
                            apply_coach_filter(retriever, pending_question)
                            response = rag_chain.invoke(
                                {"input": pending_question, "chat_history": chat_history}
                            )
                            answer = response.get("answer") or "No response generated."
                        except Exception as e:
                            _log_error("query_rag", e)
                            st.error(
                                "Something went wrong generating a response. "
                                "Please try sending your question again."
                            )
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.session_state.pending_sources = response.get("context", [])
                            if user_email:
                                save_message(user_email, st.session_state.chat_id, "assistant", answer)
                            # Rerun instead of rendering inline here. The history
                            # loop above is now the ONLY place any message is
                            # ever rendered — this used to also render here as a
                            # one-off "new" bubble, which left a second, separate
                            # on-screen position for "the latest message" that
                            # went stale across the next send: while the next
                            # answer was generating, the frontend kept showing
                            # this position's last real content (this message's
                            # own feedback buttons) until the new run finished,
                            # which looked like duplicate/premature thumbs.
                            st.rerun()