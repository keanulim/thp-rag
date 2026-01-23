import os
import json
import time
from pinecone import Pinecone, ServerlessSpec
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. INITIALIZE CLIENTS
# Using your provided keys
pc = Pinecone(api_key="pcsk_2jbfsh_8FZhtwiowGRGBfpvCQXvbeHRogdZF21raMqER6eVpwujTU64m8UMgGfBRQVtsNw")
client = genai.Client(api_key="AIzaSyDsLj0yy_yteK2ogFYgQ3_tQeEamXvG1l8")

# 2. PINECONE INDEX SETUP
INDEX_NAME = "vetted-vertical"

# Create index if it doesn't exist (Optimized for text-embedding-004)
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
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
    # Adding both your potential folders
    folders = ["Cleaned_THP_Strength", "Cleaned_John_Evans"]

    for folder in folders:
        if not os.path.exists(folder):
            print(f"📁 Folder {folder} not found, skipping...")
            continue

        # Extract coach name for metadata (e.g., "THP Strength")
        coach_name = folder.replace("Cleaned_", "").replace("_", " ")
        files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

        for filename in files:
            clean_title = filename.replace('.json', '')

            # --- EFFICIENCY CHECK: Skip if first chunk already exists in Pinecone ---
            first_chunk_id = f"{clean_title}_0"
            try:
                # We check the specific ID. If it's there, we assume the file is done.
                fetch_response = index.fetch(ids=[first_chunk_id])
                if fetch_response and first_chunk_id in fetch_response['vectors']:
                    print(f"⏩ Skipping {filename} (Already in Pinecone)")
                    continue
            except Exception:
                pass  # If fetch fails, we proceed with upload to be safe

            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            print(f"🚀 Processing: {filename}...")

            for entry in data_list:
                full_text = entry.get('cleaned_text', '')
                if not full_text:
                    continue

                # A. CHUNKING
                chunks = text_splitter.split_text(full_text)

                for i, chunk in enumerate(chunks):
                    try:
                        # B. GENERATE EMBEDDINGS
                        res = client.models.embed_content(
                            model='text-embedding-004',
                            contents=chunk,
                            config={'task_type': 'RETRIEVAL_DOCUMENT'}
                        )
                        vector = res.embeddings[0].values

                        # C. METADATA NORMALIZATION (Aligned with app.py)
                        metadata = {
                            "text": chunk,
                            "video_title": clean_title,
                            "topic": str(entry.get('primary_focus', 'Training')),
                            "difficulty": str(entry.get('difficulty', 'Intermediate')),
                            "is_injury_prevention": bool(entry.get('is_injury_prevention', False))
                        }

                        # Flatten Stats
                        stats = entry.get('stats', {})
                        if isinstance(stats, dict):
                            if stats.get('total_sets'): metadata['sets'] = stats['total_sets']
                            if stats.get('reps_per_set'): metadata['reps'] = stats['reps_per_set']
                            if stats.get('intensity_rpe'): metadata['rpe'] = stats['intensity_rpe']

                        # D. UPSERT
                        unique_id = f"{clean_title}_{i}"
                        index.upsert(vectors=[{
                            "id": unique_id,
                            "values": vector,
                            "metadata": metadata
                        }])

                        # E. RATE LIMIT SAFETY
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"❌ Error on chunk {i} of {filename}: {e}")
                        time.sleep(2)  # Back off on error

            print(f"✅ Successfully Indexed: {filename}")


if __name__ == "__main__":
    process_and_upload()
    print("\n🏀 Knowledge Base is synced! You can now test your Streamlit app.")