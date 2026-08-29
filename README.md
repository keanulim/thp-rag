# Dunk Bot

A RAG chatbot that answers vertical jump / dunk training questions by retrieving relevant clips from THP Strength and John Evans coaching video transcripts, then having Gemini answer using only that retrieved context.

## How it works

**Auth (`app.py`)**

Access is gated behind Google login, using Streamlit's built-in `st.login()` (OIDC). No passwords are handled by this app — Google does the authentication, and Streamlit stores the identity in a signed cookie in the browser. A logged-in user's email is used as their identifier for saved chats.

**Chat flow (`app.py`)**

1. You ask a question in the Streamlit UI.
2. If there's prior conversation in the current chat, a Gemini call first rewrites your question into a standalone one (so "how often should I do that?" becomes a real, searchable question using earlier context) — this is a LangChain history-aware retriever.
3. That standalone question is embedded (`gemini-embedding-001`) and used to pull the 7 most relevant transcript chunks out of a Pinecone vector index.
4. Gemini (`gemini-3.7-flash`) answers using only those chunks as context, plus the recent chat history for tone/continuity.
5. The answer is shown along with links back to the source YouTube videos (via `youtube_links.json`).

**Saved chats (`app.py` + Supabase)**

Every message is written to a `chat_messages` table in Supabase (Postgres), tagged with the user's email and a `chat_id` grouping messages into a single conversation. The sidebar lets you:

- Start a **New Chat** (generates a fresh `chat_id`)
- Browse **Previous Chats**, labeled by their first message, and reopen one
- **Delete** the currently open chat

Logging in always starts a brand-new chat — it never auto-resumes your last conversation. Supabase is accessed only from the server side using a `service_role` key (never exposed to the browser), so Row Level Security isn't required for this table.

**Data pipeline (run manually, not part of the live app)**

The chatbot's knowledge comes from a one-time (or occasional) ingestion pipeline:

1. `main.py` — scrapes video transcripts from the configured YouTube channels and saves raw `.txt` files per video.
2. `clean.py` — sends each raw transcript to Gemini to strip filler, fix punctuation, and extract structured metadata (exercises, sets/reps, RPE, etc.) as JSON.
3. `chunk.py` — wipes and rebuilds the Pinecone index, splits each cleaned transcript into overlapping chunks, embeds each chunk, and upserts it to Pinecone.
4. `script.py` — builds `youtube_links.json`, mapping each video's title to its YouTube URL so the app can link back to sources.

Re-run steps 1–4 (in order) whenever new videos should be added to the bot's knowledge base.

## Tech stack

- **UI**: Streamlit
- **Auth**: Google OAuth via Streamlit's native `st.login()`
- **LLM / embeddings**: Google Gemini (`gemini-3.7-flash` for chat, `gemini-embedding-001` for embeddings)
- **Vector store**: Pinecone (serverless)
- **Saved chats**: Supabase (Postgres)
- **Orchestration**: LangChain

## Setup

```bash
# install dependencies
uv sync   # or: pip install -e .

# add your keys
cp .env.example .env   # then fill in GOOGLE_API_KEY and PINECONE_API_KEY

# set up auth + saved chats
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in your own values

# run the app
.venv/bin/streamlit run app.py
```

Required environment variables (in `.env`, never committed):

- `GOOGLE_API_KEY` — Gemini API key
- `PINECONE_API_KEY` — Pinecone API key

Required secrets (in `.streamlit/secrets.toml`, never committed):

- `[auth]` — Google OAuth `client_id` / `client_secret` (from [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials), plus `redirect_uri` and a random `cookie_secret`
- `[supabase]` — your Supabase project `url` and `service_key` (the `service_role` key, not `anon`)

Supabase table schema:

```sql
create table chat_messages (
  id bigint generated always as identity primary key,
  user_email text not null,
  chat_id text not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);
create index chat_messages_user_email_idx on chat_messages (user_email);
create index chat_messages_chat_id_idx on chat_messages (chat_id);
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat UI + RAG chain + Google auth + saved chats |
| `main.py` | Scrapes raw YouTube transcripts |
| `clean.py` | Cleans/structures transcripts via Gemini |
| `chunk.py` | Chunks, embeds, and upserts to Pinecone |
| `script.py` | Builds `youtube_links.json` for source links |
| `query.py` | Standalone CLI script for testing the RAG chain |
