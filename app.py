import streamlit as st
import os
import json
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="THP AI")
st.title("THP AI")


@st.cache_data
def load_mappings():
    try:
        with open("youtube_links.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


url_map = load_mappings()


@st.cache_resource
def init_rag_chain():
    os.environ["GOOGLE_API_KEY"] = "API_KEY"
    os.environ["PINECONE_API_KEY"] = "API_KEY"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Pinecone(index_name="vetted-vertical", embedding=embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)

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

rag_chain = init_rag_chain()

with st.sidebar:
    st.header("Settings")
    show_debug = st.checkbox("Show Raw Context (Debug)")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

#chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a technical training question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
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
