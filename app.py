import streamlit as st
import os
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import json
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
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

    system_prompt = (
        "You are an Elite Vertical Jump Coach specializing in THP Strength and John Evans methodologies. "
        "Analyze the video transcripts provided below and give a technical answer.\n\n"
        "VIDEO CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# This function is what auditor.py will use
def query_rag(user_input: str):
    chain = init_rag_chain()
    response = chain.invoke({"input": user_input})
    return {
        "answer": response.get("answer"),
        "context": [doc.page_content for doc in response.get("context", [])]
    }


# 3. UI CODE (Browser Only)
if __name__ == "__main__":
    st.set_page_config(page_title="Dunk Bot", layout="wide")
    st.title("Dunk Bot")

    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] {
            width: fit-content;
        }
        div[class*="st-key-row-user-"] [data-testid="stChatMessage"] {
            margin-left: auto;
        }
        div[class*="st-key-row-assistant-"] [data-testid="stChatMessage"] {
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @contextmanager
    def chat_bubble(role: str, key_suffix):
        left, right = st.columns([3, 1]) if role == "assistant" else st.columns([1, 3])
        col = left if role == "assistant" else right
        with col:
            with st.container(key=f"row-{role}-{key_suffix}"):
                with st.chat_message(role, avatar=AVATARS[role]):
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

    AVATARS = {
        "user": ":material/bolt:",
        "assistant": ":material/sports_basketball:",
    }

    # --- CHAT DISPLAY ---
    for i, message in enumerate(st.session_state.messages):
        with chat_bubble(message["role"], i):
            st.markdown(message["content"])

    # --- INPUT HANDLING ---
    if user_input := st.chat_input("Ask a technical training question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_bubble("user", "new"):
            st.markdown(user_input)

        with chat_bubble("assistant", "new"):
            with st.spinner("Retrieving video data..."):
                response = rag_chain.invoke({"input": user_input})
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