import streamlit as st
import os
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import json
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

    rag_chain = init_rag_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR & DEBUGGER ---
    with st.sidebar:
        st.header("Settings")
        show_debug = st.checkbox("Show Raw Context (Debug)")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
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

        st.session_state.messages.append({"role": "assistant", "content": answer})