# Dunk Bot

A RAG chatbot that answers vertical jump / dunk training questions by retrieving relevant clips from THP Strength and John Evans coaching video transcripts, then having Gemini answer using only that retrieved context.

## How it works

**Architecture**

- `frontend/` — React + TypeScript single-page app (Vite). Talks only to `/api/*`.
- `backend/` — FastAPI server: auth, saved chats, and the RAG chain (`backend/rag.py`). Answers are streamed to the browser as newline-delimited JSON, so sources appear as soon as retrieval finishes and the answer types out as Gemini generates it.

In dev, Vite (`:5173`) proxies `/api` to FastAPI (`:8000`), so the browser sees a single origin — no CORS, and the session cookie just works.

**Auth (`backend/main.py`)**

Google login is optional, using a server-side OAuth redirect flow (Authlib) — no passwords are handled by this app; Google does the authentication, and FastAPI stores your email/name in a signed, `SameSite=Lax` session cookie. The React app never sees a token. Anyone can chat without logging in; it just isn't saved anywhere (no history, pinning, or chat titles). Logging in gives a logged-in user's email as their identifier for saved chats and unlocks that persistence.

**Chat flow (`backend/rag.py`)**

1. You ask a question in the web UI (a fresh chat starts on a hero screen with "Ask me anything about dunking").
2. If there's prior conversation in the current chat, a Gemini call first rewrites your question into a standalone one (so "how often should I do that?" becomes a real, searchable question using earlier context) — this is a LangChain history-aware retriever.
3. That standalone question is embedded (`gemini-embedding-001`). If it clearly names a coach ("John Evans" / "THP"), retrieval is constrained to that coach's videos via a Pinecone metadata filter; otherwise it searches everything. The top 15 matching transcript chunks are pulled from a Pinecone vector index.
4. Gemini (`gemini-3.7-flash`) answers using only those chunks as context (each one prefixed with its video title, coach, focus, difficulty, exercises, and stats) plus the recent chat history for tone/continuity. Chunks carry inline `[SPEAKER: ...]` tags from the cleaning step, and the system prompt is instructed to only attribute stats/results to someone if a tag or the text explicitly confirms it — otherwise it says the source is unclear rather than guessing.
5. The answer streams in, along with a collapsible list of back to the source YouTube videos — with a timestamp (`&t=Xs`) when one could be matched, via `youtube_links.json` as a fallback otherwise.

**Saved chats (`backend/store.py` + Supabase)**

Every message is written to a `chat_messages` table in Supabase (Postgres), tagged with the user's email and a `chat_id` grouping messages into a single conversation. When you send a chat's first message, Gemini generates a short title for it in parallel with the answer (stored in `chat_titles`) so the sidebar shows something more useful than a truncated snippet of what you typed. The sidebar shows your 14 most recent chats plus up to 5 pinned chats above them, each with a hover-revealed menu (⋮) to pin/unpin or delete it; **View all conversations** opens a full, searchable list of every chat. Logging in always starts a brand-new chat — it never auto-resumes your last conversation. Supabase is accessed only from the server side using a `service_role` key (never exposed to the browser), so Row Level Security isn't required for these tables.

**Data pipeline (automated daily, separate from the live app)**

The chatbot's knowledge comes from an ingestion pipeline, run automatically once a day via a macOS `launchd` job (`com.thprag.ingest`, see below) so newly-uploaded videos get added without manual intervention:

1. `main.py` — lists each channel's videos via `yt-dlp` and fetches each transcript via `youtube_transcript_api`, saving one JSON file per video (`{folder}/{video_id}.json`) with the raw timestamped segments, title, and whether the transcript was manual or auto-generated. Already-downloaded videos (by `video_id`) are skipped, so re-runs only fetch what's new. Failures (rate limits, disabled captions, etc.) are logged to `scrape_skipped.json` instead of being silently dropped. YouTube IP-blocking (`RequestBlocked`/`IpBlocked`) triggers a 2-minute backoff; if it's persistent it can take a few hours to clear.
2. `clean.py` — sends each transcript to Gemini to add punctuation/capitalization and tag speaker changes inline (`[SPEAKER: <name>]`) — it preserves the original wording rather than rewriting it, and also extracts structured metadata (focus, exercises, sets/reps/RPE, difficulty, etc.) as JSON. Already-cleaned videos are skipped.
3. `chunk.py` — splits each cleaned transcript into overlapping chunks, estimates a timestamp for each chunk by matching it back against the original transcript segments, embeds each chunk, and upserts it to Pinecone along with the extracted metadata. Videos already recorded in `chunked_files.json` are skipped, so repeat runs only embed (and bill) newly-cleaned videos — **by default it never wipes the live index**. Pass `--rebuild` to wipe and re-embed everything from scratch (e.g. after a metadata schema change); this does cause a gap in answers while it rebuilds. Embedding failures are retried and logged to `chunk_skipped.json`.
4. `script.py` — builds `youtube_links.json`, mapping each video's ID to its YouTube URL (used as a fallback source link when a chunk has no matched timestamp).

`ingest.py` runs steps 1–4 in order and is what the scheduled job invokes; run it manually (`.venv/bin/python ingest.py`) any time you want to check for new videos immediately instead of waiting for the schedule. The `launchd` job is installed at `~/Library/LaunchAgents/com.thprag.ingest.plist`, runs daily at 4:00 AM, and logs to `logs/ingest.log`. Useful commands:

```bash
# check status / see if it's currently running
launchctl print gui/$(id -u)/com.thprag.ingest

# trigger it immediately instead of waiting for 4am
launchctl kickstart gui/$(id -u)/com.thprag.ingest

# stop it from running automatically
launchctl bootout gui/$(id -u)/com.thprag.ingest
```

`query.py` is a standalone CLI for testing the exact same RAG chain the live app uses (`python query.py "your question"`).

## Tech stack

- **UI**: React + TypeScript (Vite)
- **API**: FastAPI (streaming NDJSON)
- **Auth**: Google OAuth (Authlib) + signed session cookie
- **LLM / embeddings**: Google Gemini (`gemini-3.7-flash` for chat, `gemini-embedding-001` for embeddings)
- **Vector store**: Pinecone (serverless)
- **Saved chats**: Supabase (Postgres)
- **Orchestration**: LangChain
- **Scraping**: `yt-dlp` (channel listing) + `youtube-transcript-api` (transcripts)

## Setup

```bash
# install dependencies
uv sync   # or: pip install -e .

cd frontend && npm install && cd ..

# add your keys
cp .env.example .env   # then fill in your own values

# run the API (terminal 1)
.venv/bin/uvicorn backend.main:app --reload --port 8000

# run the frontend (terminal 2) — then open http://localhost:5173
npm --prefix frontend run dev
```

For production, `npm --prefix frontend run build` outputs static files to `frontend/dist/`; serve them from the same domain as the API (e.g. a reverse proxy routing `/api` to uvicorn) and set `FRONTEND_URL`, `OAUTH_REDIRECT_URI` and `SESSION_HTTPS_ONLY=1`.

Environment variables (in `.env`, never committed — see `.env.example`):

- `GOOGLE_API_KEY` — Gemini API key
- `PINECONE_API_KEY` — Pinecone API key
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — Google OAuth (from [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials). Add `http://localhost:5173/api/auth/callback` as an Authorized redirect URI.
- `SESSION_SECRET` — random string that signs the session cookie
- `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` — your Supabase project URL and `service_role` key (not `anon`)

If these auth/Supabase values aren't in `.env`, the backend falls back to reading them from a legacy `.streamlit/secrets.toml`.

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

create table pinned_chats (
  id bigint generated always as identity primary key,
  user_email text not null,
  chat_id text not null,
  pinned_at timestamptz not null default now(),
  unique (user_email, chat_id)
);
create index pinned_chats_user_email_idx on pinned_chats (user_email);

create table chat_titles (
  chat_id text primary key,
  user_email text not null,
  title text not null,
  created_at timestamptz not null default now()
);
create index chat_titles_user_email_idx on chat_titles (user_email);
```

## Project structure

| File | Purpose |
|---|---|
| `frontend/` | React + TypeScript chat UI (Vite) |
| `backend/main.py` | FastAPI routes: Google auth, saved chats, streaming `/api/chat` |
| `backend/rag.py` | Topic gate, coach filter, retrieval chain, chat titles |
| `backend/store.py` | Supabase persistence (messages, titles, pins) |
| `backend/config.py` | Settings from `.env` |
| `ingest.py` | Runs the full pipeline (steps 1–4 below) in order; invoked daily by `launchd` |
| `main.py` | Scrapes raw YouTube transcripts (channel listing + captions) |
| `clean.py` | Punctuates transcripts and tags speakers via Gemini |
| `chunk.py` | Chunks, timestamps, embeds, and upserts to Pinecone |
| `script.py` | Builds `youtube_links.json` for fallback source links |
| `query.py` | Standalone CLI for testing the live RAG chain |
