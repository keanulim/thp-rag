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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
@st.cache_resource
def init_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="retrieval_query",
    )
    vectorstore = Pinecone(index_name="vetted-vertical", embedding=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

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
        "VIDEO CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


# This function is what auditor.py will use
def query_rag(user_input: str, chat_history: list | None = None):
    chain = init_rag_chain()
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
            border-color: #2E2E2E !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: #5C5C5C !important;
            font-weight: 300 !important;
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

    rag_chain = init_rag_chain()

    # A fresh chat_id each time the session starts (e.g. on login) — never
    # auto-resumes a previous conversation.
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []

    # --- SIDEBAR & DEBUGGER ---
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Signed in as {st.user.email}")
        if st.button("Log out"):
            st.logout()
        show_debug = st.checkbox("Show Raw Context (Debug)")

        st.divider()
        if st.button("+ New Chat", use_container_width=True):
            st.session_state.chat_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages and st.button("Delete This Chat", use_container_width=True):
            delete_chat(user_email, st.session_state.chat_id)
            st.session_state.chat_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption("Previous Chats")
        for chat in list_chats(user_email):
            is_current = chat["chat_id"] == st.session_state.chat_id
            label = chat["title"] or "(empty chat)"
            if st.button(label, key=f"chat-{chat['chat_id']}", use_container_width=True,
                         disabled=is_current):
                st.session_state.chat_id = chat["chat_id"]
                st.session_state.messages = load_history(user_email, chat["chat_id"])
                st.rerun()

    # --- CHAT DISPLAY ---
    for i, message in enumerate(st.session_state.messages):
        with chat_bubble(message["role"], i):
            st.markdown(message["content"])

    # --- INPUT HANDLING ---
    if user_input := st.chat_input("Ask a technical training question..."):
        chat_history = [
            ("human" if m["role"] == "user" else "ai", m["content"])
            for m in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(user_email, st.session_state.chat_id, "user", user_input)
        with chat_bubble("user", "new"):
            st.markdown(user_input)

        with chat_bubble("assistant", "new"):
            with st.spinner("Retrieving video data..."):
                response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
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
                        url = url_map.get(title, "https://www.youtube.com/@thpstrength1/videos")
                        st.markdown(f"🔗 [{title}]({url})")

        save_message(user_email, st.session_state.chat_id, "assistant", answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})