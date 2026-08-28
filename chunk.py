import os
import json
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. INITIALIZE CLIENTS
load_dotenv()

# 3. INITIALIZE CLIENTS USING ENV KEYS
# These names must match what you wrote inside your .env file
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "vetted-vertical"

pc = Pinecone(api_key=PINECONE_KEY)
client = genai.Client(api_key=GEMINI_KEY)

# 2. THE CLEAN SLATE (Crucial Step)
if INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
    print(f"🗑️ Wiping existing index: {INDEX_NAME}...")
    pc.delete_index(INDEX_NAME)

print(f"🏗️ Creating fresh index: {INDEX_NAME}...")
pc.create_index(
    name=INDEX_NAME,
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index(INDEX_NAME)

# 3. LANGCHAIN SMART SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)


def process_and_upload():
    folders = ["Cleaned_THP_Strength", "Cleaned_John_Evans"]

    for folder in folders:
        if not os.path.exists(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

        for filename in files:
            clean_title = filename.replace('.json', '')
            file_path = os.path.join(folder, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            print(f"🚀 Indexing: {filename}...")

            for entry in data_list:
                full_text = entry.get('cleaned_text', '')
                chunks = text_splitter.split_text(full_text)

                for i, chunk in enumerate(chunks):
                    try:
                        # --- THE FIX: USE THE SAME MODEL AS APP.PY ---
                        res = client.models.embed_content(
                            model='gemini-embedding-001',
                            contents=chunk,
                            config={'task_type': 'RETRIEVAL_DOCUMENT'},

                        )
                        vector = res.embeddings[0].values

                        metadata = {
                            "text": chunk,
                            "video_title": clean_title,
                            "coach": folder.split('_')[1]  # "THP" or "John"
                        }

                        index.upsert(vectors=[{
                            "id": f"{clean_title}_{i}",
                            "values": vector,
                            "metadata": metadata
                        }])

                        time.sleep(0.1)  # Respect Rate Limits

                    except Exception as e:
                        print(f"❌ Error on {filename}: {e}")
                        time.sleep(2)


if __name__ == "__main__":
    process_and_upload()
    print("\n🏀 Knowledge Base is synced! Now run your auditor.")