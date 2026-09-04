import os
import re
import json
import time
import httpx
from difflib import SequenceMatcher
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from langchain_text_splitters import RecursiveCharacterTextSplitter

REQUEST_TIMEOUT_MS = 120_000
EMBED_BATCH_SIZE = 100  # Google's max texts per embed_content call

SPEAKER_TAG_RE = re.compile(r'^\[SPEAKER:[^\]]*\]\s*')
SKIP_LOG = "chunk_skipped.json"
CHUNKED_STATE_FILE = "chunked_files.json"

# 1. INITIALIZE CLIENTS
load_dotenv()

# 3. INITIALIZE CLIENTS USING ENV KEYS
# These names must match what you wrote inside your .env file
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "vetted-vertical"

pc = Pinecone(api_key=PINECONE_KEY)
client = genai.Client(api_key=GEMINI_KEY)


# 2. THE CLEAN SLATE (Crucial Step) — deliberately NOT run at import time.
# Importing this module (e.g. to reuse a helper elsewhere) must never wipe
# the live index as a side effect; only running it as a script should.
def setup_index():
    """Wipe (if present) and recreate the index, then return a fresh client
    handle to it. pc.Index(name) resolves the index's host from the control
    plane at construction time — it must be constructed AFTER the index is
    (re)created, not before, or it'll hold a stale/nonexistent host."""
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

    return pc.Index(INDEX_NAME)


def load_skip_log():
    if os.path.exists(SKIP_LOG):
        with open(SKIP_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def log_skip(skipped, video_id, chunk_index, reason):
    skipped.append({"video_id": video_id, "chunk_index": chunk_index, "reason": reason})
    with open(SKIP_LOG, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)


def load_chunked_files() -> set[str]:
    """Cleaned_*/filename.json keys already embedded and upserted into the
    live index — lets repeat runs only process newly-cleaned videos instead
    of re-embedding (and re-billing) everything each time."""
    if os.path.exists(CHUNKED_STATE_FILE):
        with open(CHUNKED_STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def mark_chunked(chunked_files: set[str], key: str):
    chunked_files.add(key)
    with open(CHUNKED_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(chunked_files), f, indent=2)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ClientError, httpx.TimeoutException))
)
def embed_batch(chunks: list[str]):
    """Embed up to EMBED_BATCH_SIZE chunks in a single API call instead of
    one call per chunk — the same total token volume, but a fraction of the
    requests, which is what was actually making this step slow."""
    return client.models.embed_content(
        model='gemini-embedding-001',
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type='RETRIEVAL_DOCUMENT',
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        ),
    )


def embed_all(chunks: list[str]):
    """Embed an arbitrary number of chunks, splitting into API-size-limited
    batches as needed. Returns one embedding vector per input chunk, in order."""
    vectors = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        res = embed_batch(batch)
        vectors.extend(e.values for e in res.embeddings)
    return vectors


# 3. LANGCHAIN SMART SPLITTER
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)


def build_time_index(segments):
    """Join raw transcript segments into one string, remembering each
    segment's [start_offset, end_offset) span in that string and its
    [start, start+duration) time span — lets a chunk of *cleaned* text be
    matched back to an approximate time, interpolated within a segment."""
    raw_parts = []
    index = []  # (start_offset, end_offset, seg_start, seg_duration, seg_text)
    cursor = 0
    for seg in segments:
        text = seg.get("text", "")
        start_offset = cursor
        end_offset = cursor + len(text)
        index.append((start_offset, end_offset, seg.get("start", 0.0), seg.get("duration", 0.0), text))
        raw_parts.append(text)
        cursor = end_offset + 1  # +1 for the space " ".join will add
    return " ".join(raw_parts), index


def offset_to_timestamp(offset, time_index):
    """Timestamp at a raw_text offset, interpolated by word position within
    whichever segment contains it (words are a better proxy for speaking
    pace than raw character count)."""
    if not time_index:
        return 0.0
    if offset < time_index[0][0]:
        return time_index[0][2]

    for start_offset, end_offset, seg_start, seg_duration, seg_text in time_index:
        if start_offset <= offset <= end_offset:
            local_pos = offset - start_offset
            words_before = seg_text[:local_pos].split()
            total_words = seg_text.split()
            frac = len(words_before) / len(total_words) if total_words else 0.0
            return seg_start + frac * seg_duration

    last_start_offset, last_end_offset, last_seg_start, last_seg_duration, _ = time_index[-1]
    return last_seg_start + last_seg_duration


def estimate_chunk_timestamp(chunk_text, chunk_position, cleaned_text_len, raw_text, time_index, total_duration):
    """Best-effort timestamp for a chunk of Gemini-cleaned text.

    Primary: locate the chunk's opening words in the raw, timestamped
    transcript — exact substring match first (most precise), falling back
    to fuzzy matching (cleaning rewords things, so exact often won't hit).
    Interpolates within the matched segment rather than snapping to its
    start. Falls back to a proportional video-position estimate only if no
    confident match is found at all.
    """
    needle = SPEAKER_TAG_RE.sub('', chunk_text).strip()[:60]

    if needle and raw_text:
        exact_pos = raw_text.find(needle)
        if exact_pos != -1:
            return round(offset_to_timestamp(exact_pos, time_index), 1)

        matcher = SequenceMatcher(None, raw_text, needle, autojunk=False)
        match = matcher.find_longest_match(0, len(raw_text), 0, len(needle))
        if match.size >= max(15, len(needle) * 0.5):
            return round(offset_to_timestamp(match.a, time_index), 1)

    if cleaned_text_len:
        return round((chunk_position / cleaned_text_len) * total_duration, 1)
    return 0.0


def flatten_extracted_metadata(entry):
    """Coaching metadata clean.py extracts (focus, exercises, stats, taxonomy)
    was previously discarded here entirely — never stored, never reaching the
    LLM. Kept as Pinecone-safe scalars with "N/A" placeholders so every chunk
    has the same keys (document_prompt in app.py requires that)."""
    stats = entry.get('stats') or {}
    taxonomy = entry.get('movement_taxonomy') or {}

    def s(value):
        return str(value) if value not in (None, "") else "N/A"

    exercises = entry.get("exercise_list") or []

    stats_parts = []
    if stats.get("reps_per_set") is not None:
        stats_parts.append(f"{stats['reps_per_set']} reps")
    if stats.get("total_sets") is not None:
        stats_parts.append(f"{stats['total_sets']} sets")
    if stats.get("intensity_rpe") is not None:
        stats_parts.append(f"RPE {stats['intensity_rpe']}")

    return {
        "primary_focus": s(entry.get("primary_focus")),
        "exercise_list": ", ".join(exercises) if exercises else "N/A",
        "difficulty": s(entry.get("difficulty")),
        "stats_summary": " x ".join(stats_parts) if stats_parts else "N/A",
        "is_injury_prevention": bool(entry.get("is_injury_prevention")),
        "jump_type": s(taxonomy.get("jump_type")),
        "plant_foot": s(taxonomy.get("plant_foot")),
        "body_part": taxonomy.get("body_part") or [],
    }


def process_and_upload(index):
    folders = ["Cleaned_THP_Strength", "Cleaned_John_Evans"]
    skipped = load_skip_log()
    chunked_files = load_chunked_files()

    for folder in folders:
        if not os.path.exists(folder):
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

        for filename in files:
            state_key = f"{folder}/{filename}"
            if state_key in chunked_files:
                continue

            file_path = os.path.join(folder, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)

            print(f"🚀 Indexing: {filename}...")

            for entry_idx, entry in enumerate(data_list):
                full_text = entry.get('cleaned_text', '')
                chunks = text_splitter.split_text(full_text)
                video_id = entry.get('video_id', filename.replace('.json', ''))
                video_title = entry.get('video_title', video_id)

                segments = entry.get('segments', [])
                raw_text, time_index = build_time_index(segments)
                total_duration = (
                    segments[-1]["start"] + segments[-1]["duration"] if segments else 0.0
                )
                extracted_metadata = flatten_extracted_metadata(entry)

                if not chunks:
                    continue

                try:
                    # One batched call (or a few, if over EMBED_BATCH_SIZE)
                    # for every chunk in this video, instead of one call per
                    # chunk — same total tokens, far fewer requests.
                    vectors = embed_all(chunks)

                    pinecone_vectors = []
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                        chunk_position = max(full_text.find(chunk), 0)
                        start_time = estimate_chunk_timestamp(
                            chunk, chunk_position, len(full_text), raw_text, time_index, total_duration
                        )
                        metadata = {
                            "text": chunk,
                            "video_id": video_id,
                            "video_title": video_title,
                            "coach": folder.split('_')[1],  # "THP" or "John"
                            "start_time": start_time,
                            **extracted_metadata,
                        }
                        pinecone_vectors.append({
                            "id": f"{video_id}_{entry_idx}_{i}",
                            "values": vector,
                            "metadata": metadata,
                        })

                    index.upsert(vectors=pinecone_vectors)
                    time.sleep(0.2)  # Respect rate limits between videos

                except Exception as e:
                    print(f"❌ Error on {filename}: {e}")
                    log_skip(skipped, video_id, -1, str(e)[:200])
                    time.sleep(2)

            mark_chunked(chunked_files, state_key)


def get_or_create_index():
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"🏗️ Creating index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)


if __name__ == "__main__":
    import sys

    # Default: incremental — only embeds/upserts videos not already recorded
    # in chunked_files.json, and never touches existing vectors. Pass
    # --rebuild to wipe the index and re-embed everything from scratch (e.g.
    # after a metadata schema change).
    if "--rebuild" in sys.argv:
        live_index = setup_index()
        if os.path.exists(CHUNKED_STATE_FILE):
            os.remove(CHUNKED_STATE_FILE)
    else:
        live_index = get_or_create_index()

    process_and_upload(live_index)
    print("\n🏀 Knowledge Base is synced! Now run your auditor.")