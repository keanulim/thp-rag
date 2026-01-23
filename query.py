from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic import hub
import os

# Set this once at the top - LangChain components will find it automatically
os.environ["GOOGLE_API_KEY"] = "AIzaSyDsLj0yy_yteK2ogFYgQ3_tQeEamXvG1l8"
os.environ["PINECONE_API_KEY"] = "pcsk_2jbfsh_8FZhtwiowGRGBfpvCQXvbeHRogdZF21raMqER6eVpwujTU64m8UMgGfBRQVtsNw"
# 1. Initialize the Brain & Ears
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

# 2. Connect to the Index you just built
vectorstore = Pinecone(index_name="vetted-vertical", embedding=embeddings)

# 3. Create the Retrieval Chain
# This pulls the top 3 most relevant chunks from Pinecone
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 4. Ask a Vetted Question
response = rag_chain.invoke({"input": "how many sets should i do for each exercise?"})

print(response["answer"])