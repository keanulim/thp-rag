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


# 1b. PER-USER CHAT PERSISTENCE (Supabase)
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["service_key"])


def load_history(email: str, chat_id: str) -> list[dict]:
    res = (
        get_supabase()
        .table("chat_messages")
        .select("role, content")
        .eq("user_email", email)
        .eq("chat_id", chat_id)
        .order("id")
        .execute()
    )
    return res.data or []


def save_message(email: str, chat_id: str, role: str, content: str):
    get_supabase().table("chat_messages").insert(
        {"user_email": email, "chat_id": chat_id, "role": role, "content": content}
    ).execute()


def delete_chat(email: str, chat_id: str):
    get_supabase().table("chat_messages").delete().eq("user_email", email).eq("chat_id", chat_id).execute()


def list_chats(email: str) -> list[dict]:
    res = (
        get_supabase()
        .table("chat_messages")
        .select("chat_id, role, content, created_at")
        .eq("user_email", email)
        .order("created_at")
        .execute()
    )
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
    return sorted(chats.values(), key=lambda c: c["started_at"], reverse=True)


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
            border-radius: 22px;
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
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] div:has(> [data-testid="stChatInputTextArea"]),
        [data-testid="stChatInput"] div:has(> div > [data-testid="stChatInputTextArea"]) {
            background-color: transparent !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: #5C5C5C !important;
            font-weight: 300 !important;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 8px;
            transition: background-color 0.15s ease;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] > div {
            justify-content: flex-start;
        }
        div[class*="st-key-chat-"] button[data-testid="stBaseButton-secondary"] p {
            text-align: left;
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
        [data-testid="stSidebar"] {
            background-color: #121212 !important;
            border-right: 0.5px solid #333333;
            min-width: 260px !important;
        }
        [data-testid="stSidebar"] button p {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .st-key-sidebar-settings [data-testid="stCaptionContainer"] {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .st-key-previous-chats-list {
            gap: 0 !important;
        }
        div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            min-height: 0 !important;
            padding: 4px !important;
            opacity: 0 !important;
            transition: opacity 0.15s ease, background-color 0.15s ease;
            position: relative;
            top: 8.3125px;
        }
        div[class*="st-key-menu-"] button[data-testid="stBaseButton-secondary"]:hover {
            background: rgba(255, 255, 255, 0.15) !important;
        }
        div[class*="st-key-menu-"] [data-testid="stIconMaterial"] {
            font-size: 16px !important;
        }
        div[class*="st-key-chatrow-"] {
            border-radius: 8px;
            transition: background-color 0.15s ease;
        }
        div[class*="st-key-menuwrap-"] {
            position: relative;
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    @contextmanager
    def chat_bubble(role: str, key_suffix):
        with st.container(key=f"row-{role}-{key_suffix}"):
            with st.chat_message(role):
                yield

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

    if not st.user.is_logged_in:
        st.title("Dunk Bot")
        st.write("Sign in with Google to start chatting.")
        if st.button("Log in with Google"):
            st.login()
        st.stop()

    user_email = st.user.email
    first_name = st.user.get("given_name") or st.user.get("name") or "Your"

    rag_chain, retriever = init_rag_chain()

    # A fresh chat_id each time the session starts (e.g. on login) — never
    # auto-resumes a previous conversation.
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []

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
        st.caption(f"{first_name}'s Chats")
        with st.container(key="previous-chats-list"):
            all_chats = list_chats(user_email)
            for chat in all_chats[:15]:
                is_current = chat["chat_id"] == st.session_state.chat_id
                label = chat["title"] or "(empty chat)"
                with st.container(key=f"chatrow-{chat['chat_id']}"):
                    title_col, menu_col = st.columns([6, 1], vertical_alignment="center")
                    with title_col:
                        if st.button(label, key=f"chat-{chat['chat_id']}", use_container_width=True,
                                     disabled=is_current):
                            st.session_state.chat_id = chat["chat_id"]
                            st.session_state.messages = load_history(user_email, chat["chat_id"])
                            st.session_state.view = "chat"
                            st.rerun()
                    with menu_col:
                        # Menu visibility is handled purely by CSS (:focus-within on
                        # menuwrap below) — clicking elsewhere blurs the trigger and
                        # closes it natively, no session_state toggle needed.
                        with st.container(key=f"menuwrap-{chat['chat_id']}"):
                            st.button("", key=f"menu-{chat['chat_id']}", icon=":material/more_vert:")
                            with st.container(key=f"menucontent-{chat['chat_id']}"):
                                if st.button("Delete", key=f"delete-{chat['chat_id']}", icon=":material/delete:",
                                             use_container_width=True):
                                    delete_chat(user_email, chat["chat_id"])
                                    if is_current:
                                        st.session_state.chat_id = str(uuid.uuid4())
                                        st.session_state.messages = []
                                    st.rerun()

            if len(all_chats) > 15:
                if st.button("See more", key="see-more-chats"):
                    st.session_state.view = "all_chats"
                    st.rerun()

        with st.container(key="sidebar-settings"):
            st.divider()
            st.header("Settings")
            st.caption(f"Signed in as {st.user.email}")
            if st.button("Log out"):
                st.logout()
            show_debug = st.checkbox("Show Raw Context (Debug)")

    # --- ALL CHATS (SEARCH) PAGE ---
    if st.session_state.view == "all_chats":
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
                st.session_state.messages = load_history(user_email, chat["chat_id"])
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
                # on a rerun — a rerun only prunes it once the new run finishes,
                # which left it on screen for the whole (slow) RAG call.
                hero_placeholder.empty()
                st.session_state.messages.append({"role": "user", "content": hero_input})
                save_message(user_email, st.session_state.chat_id, "user", hero_input)
                is_new_chat = False

        if not is_new_chat:
            for i, message in enumerate(st.session_state.messages):
                with chat_bubble(message["role"], i):
                    st.markdown(message["content"])

            user_input = st.chat_input("Ask a technical training question...", key="chat_input_bottom")

            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
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
                        apply_coach_filter(retriever, pending_question)
                        response = rag_chain.invoke({"input": pending_question, "chat_history": chat_history})
                        answer = response.get("answer", "No response generated.")

                        if show_debug:
                            with st.expander("🔬 Raw Pinecone Chunks"):
                                for i, doc in enumerate(response.get("context", [])):
                                    st.write(f"**Chunk {i + 1} from {doc.metadata.get('video_title')}**")
                                    st.info(doc.page_content)

                        st.markdown(answer)

                        with st.expander("Sources & References"):
                            for doc in response.get("context", []):
                                title = doc.metadata.get("video_title", "Untitled Video")
                                video_id = doc.metadata.get("video_id")
                                start_time = doc.metadata.get("start_time")

                                if video_id and start_time is not None:
                                    url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                                else:
                                    url = url_map.get(video_id, "https://www.youtube.com/@thpstrength1/videos")

                                st.markdown(f"🔗 [{title}]({url})")

                save_message(user_email, st.session_state.chat_id, "assistant", answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})