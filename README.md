# Dunk Bot

A RAG chatbot that answers vertical jump / dunk training questions by retrieving relevant clips from THP Strength and John Evans coaching video transcripts, then having Gemini answer using only that retrieved context.

## How it works

**Chat flow (`app.py`)**

1. You ask a question in the Streamlit UI.
2. If there's prior conversation in the session, a Gemini call first rewrites your question into a standalone one (so "how often should I do that?" becomes a real, searchable question using earlier context) — this is a LangChain history-aware retriever.
3. That standalone question is embedded (`gemini-embedding-001`) and used to pull the 7 most relevant transcript chunks out of a Pinecone vector index.
4. Gemini (`gemini-3.7-flash`) answers using only those chunks as context, plus the recent chat history for tone/continuity.
5. The answer is shown along with links back to the source YouTube videos (via `youtube_links.json`).

Chat history lives only in Streamlit's session state — nothing is persisted between browser sessions or written to disk.

**Data pipeline (run manually, not part of the live app)**

The chatbot's knowledge comes from a one-time (or occasional) ingestion pipeline:

1. `main.py` — scrapes video transcripts from the configured YouTube channels and saves raw `.txt` files per video.
2. `clean.py` — sends each raw transcript to Gemini to strip filler, fix punctuation, and extract structured metadata (exercises, sets/reps, RPE, etc.) as JSON.
3. `chunk.py` — wipes and rebuilds the Pinecone index, splits each cleaned transcript into overlapping chunks, embeds each chunk, and upserts it to Pinecone.
4. `script.py` — builds `youtube_links.json`, mapping each video's title to its YouTube URL so the app can link back to sources.

Re-run steps 1–4 (in order) whenever new videos should be added to the bot's knowledge base.

## Tech stack

- **UI**: Streamlit
- **LLM / embeddings**: Google Gemini (`gemini-3.7-flash` for chat, `gemini-embedding-001` for embeddings)
- **Vector store**: Pinecone (serverless)
- **Orchestration**: LangChain

## Setup

```bash
# install dependencies
uv sync   # or: pip install -e .

# add your keys
cp .env.example .env   # then fill in GOOGLE_API_KEY and PINECONE_API_KEY

# run the app
.venv/bin/streamlit run app.py
```

Required environment variables (in `.env`, never committed):

- `GOOGLE_API_KEY` — Gemini API key
- `PINECONE_API_KEY` — Pinecone API key

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI + RAG chain |
| `main.py` | Scrapes raw YouTube transcripts |
| `clean.py` | Cleans/structures transcripts via Gemini |
| `chunk.py` | Chunks, embeds, and upserts to Pinecone |
| `script.py` | Builds `youtube_links.json` for source links |
| `query.py` | Standalone CLI script for testing the RAG chain |
