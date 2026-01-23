from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
import os

os.environ["GOOGLE_API_KEY"] = "API_KEY"
os.environ["PINECONE_API_KEY"] = "API_KEY"

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

vectorstore = Pinecone(index_name="vetted-vertical", embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = rag_chain.invoke({"input": "whats the best exercise to increase my vertical jump?"})

print(response["answer"])
