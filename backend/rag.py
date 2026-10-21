"""The RAG engine: topic gate, coach filter, retrieval chain, title generation.

No web framework imports here — used by both the API (backend/main.py) and the
CLI (query.py).
"""
import json
import logging
from functools import lru_cache
from typing import Iterator

from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from .config import ROOT

ChatHistory = list[tuple[str, str]]  # [("human" | "ai", content), ...]

DEFAULT_SOURCE_URL = "https://www.youtube.com/@thpstrength1/videos"


@lru_cache
def load_url_map() -> dict[str, str]:
    try:
        with open(ROOT / "youtube_links.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _text(content) -> str:
    """Gemini can return content as a list of parts instead of a plain string."""
    if isinstance(content, list):
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    return content or ""


# --- Chat titles ---

@lru_cache
def get_title_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3, max_output_tokens=30)


def generate_chat_title(question: str) -> str:
    fallback = question[:40] + ("…" if len(question) > 40 else "")
    try:
        prompt = (
            "You're titling conversations in a vertical jump / dunk training chatbot, so every "
            "message is already about that topic — don't restate it. Reply with exactly ONE "
            "short chat title (3-6 words) capturing what's specifically being asked in this "
            "message. Output only that single title — no alternatives, no bullets, no quotes, "
            "no trailing punctuation:\n\n"
            f"{question}"
        )
        raw_content = _text(get_title_llm().invoke(prompt).content)
        # Guard against the model still returning multiple lines/bullets despite
        # the "exactly ONE title" instruction — take just the first non-empty one.
        title = next((line.strip() for line in raw_content.splitlines() if line.strip()), "")
        title = title.lstrip("*-•").strip().strip('"').strip()
        return title[:60] if title else fallback
    except Exception as e:
        logging.getLogger("dunkbot").warning("title generation failed: %r", e)
        return fallback


# --- Topic gate ---

# Only THP Strength and John Evans are actual channels/coaches — their videos
# are tagged with a "coach" metadata field ("THP" or "John") that retrieval
# can filter on. Guests below aren't tied to one channel, so they're handled
# via the [SPEAKER: ...] tags in the text instead, not a retrieval filter.
KNOWN_GUESTS = [
    "Josh Ruble",
    "Dom Gonzales (aka Dom Dunks)",
    "Donovan Hawkins",
    "Austin",
    "Ben Moxness",
]

OFF_TOPIC_MESSAGE = (
    "I'm built specifically for vertical jump and dunk training — ask me about plyometrics, "
    "squat technique, jump programming, or anything from THP Strength / John Evans and I'll "
    "dig into the video library for you."
)


def infer_coach_filter(question: str) -> str | None:
    """Best-effort guess at which channel's videos to constrain retrieval to."""
    q = question.lower()
    if "evans" in q:
        return "John"
    if "thp" in q:
        return "THP"
    return None


@lru_cache
def get_topic_classifier_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_output_tokens=5)


def is_dunking_related(question: str, chat_history: ChatHistory | None = None) -> bool:
    """Cheap gate to skip Pinecone retrieval entirely for obviously off-topic messages."""
    history_preview = ""
    if chat_history:
        history_preview = "\n".join(f"{role}: {content}" for role, content in chat_history[-4:])
    prompt = (
        "You're a gate in front of a chatbot that ONLY answers questions about basketball "
        "dunking, vertical jump training, plyometrics, strength/speed training, or coaching "
        "content from THP Strength and John Evans. Given the conversation so far and the newest "
        "message, decide if that newest message is on-topic for this chatbot — including vague "
        "follow-ups (e.g. 'why', 'how often should I do that') that make sense given the context. "
        "Reply with exactly one word: YES or NO.\n\n"
        + (f"Recent conversation:\n{history_preview}\n\n" if history_preview else "")
        + f"Newest message: {question}"
    )
    try:
        result = _text(get_topic_classifier_llm().invoke(prompt).content)
        return not result.strip().upper().startswith("N")
    except Exception:
        return True  # fail open -- never block a legit question over a classifier hiccup


# --- Retrieval chain ---

def format_timestamp(seconds) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def source_url(video_id: str | None, start_time) -> str:
    if video_id and start_time is not None:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
    return load_url_map().get(video_id, DEFAULT_SOURCE_URL)


def add_timestamp_display(docs):
    for doc in docs:
        title = doc.metadata.get("video_title", "Untitled Video")
        start_time = doc.metadata.get("start_time")
        timestamp_display = format_timestamp(start_time)
        doc.metadata["timestamp_display"] = timestamp_display
        url = source_url(doc.metadata.get("video_id"), start_time)
        doc.metadata["citation_link"] = f"[{title}, {timestamp_display}]({url})"
    return docs


def load_knowledge() -> str:
    """Coach-curated defaults (backend/knowledge/*.md), injected into every prompt.

    Braces are escaped because the system prompt is a LangChain template,
    where {name} means a variable.
    """
    parts = [p.read_text() for p in sorted((ROOT / "backend" / "knowledge").glob("*.md"))]
    return "\n\n".join(parts).replace("{", "{{").replace("}", "}}")


SYSTEM_PROMPT = (
    "You are an Elite Vertical Jump Coach specializing in THP Strength and John Evans methodologies. "
    "Analyze the video transcripts provided below and give a technical answer.\n\n"
    "Some clips feature a guest or athlete speaking in first person about their own training or "
    "results, not the channel's coach — recurring guests include: " + ", ".join(KNOWN_GUESTS) + ". "
    "Each chunk of context below is prefixed with its video title and coach/channel — use that to "
    "tell apart content from different videos or sources. The text may also contain inline tags "
    "like '[SPEAKER: <name>]' marking who is talking in the text that follows. Only attribute a "
    "specific person's stats, results, or experiences to them if a [SPEAKER: ...] tag or explicit "
    "statement in the text confirms it — match names/aliases above to the same person. If the "
    "relevant text is tagged [SPEAKER: Uncertain], has no speaker tag, or you otherwise can't tell "
    "whose stats or experience is being described, say so instead of guessing which coach or "
    "athlete it refers to.\n\n"
    "FORMATTING: Do not bold the lead-in phrase of a bullet OR numbered list item. This applies "
    "to every list in the answer, not just some of them. Do NOT write "
    "'* **Tendon response:** the tendon adapts...' or '1. **Order First:** do X...' — write "
    "'* The tendon adapts...' or '1. Do X first...' instead. List items must be plain text with "
    "no bolded words anywhere in them, including inside parentheticals. Reserve **bold** only for "
    "a single specific number or term in ordinary paragraph text (never inside a list) that's "
    "genuinely easy to miss, and only if truly necessary — most answers should have zero bolded "
    "text. Before finalizing, check every list item for a leading **bold** span and rewrite it "
    "as plain text if found.\n\n"
    "QUOTES & TIMESTAMPS: Each context chunk below is tagged with a Video title, Timestamp "
    "(MM:SS or H:MM:SS marking where that chunk starts), and a Citation Link — a ready-made "
    "markdown link like '[How to Squat Properly, 4:12](https://www.youtube.com/watch?v=...)'. "
    "For the most load-bearing claim, cue, or number in your answer (one or two in a short answer, "
    "up to three or four in a long one), back it with a "
    "short direct quote (a few words to one sentence, copied verbatim from that chunk's text) "
    "followed immediately by that chunk's exact Citation Link, copied character-for-character "
    "— e.g. — \"raise the hips before the shoulders\" [How to Squat Properly, 4:12](https://www."
    "youtube.com/watch?v=abc123&t=252s). Never retype, shorten, or hand-construct the link "
    "yourself — paste the given Citation Link markdown exactly as-is, right after the quote it "
    "supports. Not every sentence needs a quote; use them where a direct quote adds real "
    "credibility, not as decoration on every line.\n\n"
    "LENGTH: Match the answer's length to the question. Lead with the answer itself; no intro "
    "restating the question and no closing summary. A simple question (how many sets, how long to "
    "rest, is X worth doing) gets a direct answer in one to three sentences. An explanation gets a "
    "short paragraph or a few bullets. Only a program or a genuinely multi-part question gets a long, "
    "structured answer, and even then keep each list item short (the exercise and its sets/reps, not "
    "a justification for it). If there's real nuance, such as a condition that changes the answer, "
    "mention it briefly in a sentence; don't explain the underlying physiology or reasoning unless "
    "asked. Use headings only in long answers.\n\n"
    "DEFAULT ANSWERS: The coach-curated knowledge below defines what a typical answer looks like. "
    "For general questions it covers (e.g. 'write me a program', 'how should I structure my week'), "
    "your answer must follow its default structure, filling in specific exercise choices, cues, and "
    "sets/reps from the video context and citing them. Only switch to one of its variants when the "
    "user states or clearly implies that variant's condition, and say which condition you're adapting "
    "for. Do not build a general answer around a specialized protocol, a single athlete's custom "
    "program, or a program for another sport just because it appears in the video context; mention "
    "those only when the user asks about them.\n\n"
    "COACH KNOWLEDGE:\n" + load_knowledge() + "\n\n"
    "VIDEO CONTEXT:\n{context}"
)


@lru_cache
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="retrieval_query")
    return Pinecone(index_name="vetted-vertical", embedding=embeddings)


@lru_cache
def get_rag_chain(coach: str | None):
    """One immutable chain per coach filter (None / "John" / "THP").

    The Streamlit version mutated a single shared retriever's search_kwargs
    per query; under a concurrent web server that lets one request's filter
    leak into another's, so each filter gets its own chain instead.
    """
    search_kwargs: dict = {"k": 15}
    if coach:
        search_kwargs["filter"] = {"coach": coach}
    retriever = get_vectorstore().as_retriever(search_kwargs=search_kwargs)
    retriever_with_timestamps = retriever | RunnableLambda(add_timestamp_display)

    llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.2)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rephrase the "
         "question into a standalone question that can be understood without "
         "the chat history. Do not answer it, just reformulate it if needed "
         "and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever_with_timestamps, contextualize_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_prompt = PromptTemplate.from_template(
        "[Video: {video_title} | Coach: {coach} | Timestamp: {timestamp_display} | "
        "Citation Link: {citation_link} | Focus: {primary_focus} | Difficulty: {difficulty} | "
        "Exercises: {exercise_list} | Stats: {stats_summary}]\n"
        "{page_content}"
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)


def dedupe_sources(docs) -> list[dict]:
    """One entry per video, in retrieval order, shaped for the frontend."""
    seen: set = set()
    sources = []
    for doc in docs:
        video_id = doc.metadata.get("video_id")
        if video_id in seen:
            continue
        seen.add(video_id)
        start_time = doc.metadata.get("start_time")
        sources.append({
            "title": doc.metadata.get("video_title", "Untitled Video"),
            "coach": {"THP": "THP Strength", "John": "John Evans"}.get(doc.metadata.get("coach"), ""),
            "timestamp": format_timestamp(start_time),
            "url": source_url(video_id, start_time),
        })
    return sources


def stream_answer(question: str, chat_history: ChatHistory) -> Iterator[dict]:
    """Yields {"type": "sources", ...} once, then {"type": "token", ...} pieces.

    create_retrieval_chain streams a dict per step: the retrieved "context"
    arrives whole (it isn't a token stream), then "answer" arrives in pieces
    as Gemini generates it.
    """
    if not is_dunking_related(question, chat_history):
        yield {"type": "sources", "sources": []}
        yield {"type": "token", "text": OFF_TOPIC_MESSAGE}
        return
    chain = get_rag_chain(infer_coach_filter(question))
    for chunk in chain.stream({"input": question, "chat_history": chat_history}):
        if "context" in chunk:
            yield {"type": "sources", "sources": dedupe_sources(chunk["context"])}
        if "answer" in chunk:
            text = _text(chunk["answer"])
            if text:
                yield {"type": "token", "text": text}


def query_rag(user_input: str, chat_history: ChatHistory | None = None):
    """Non-streaming call, used by the query.py CLI."""
    if not is_dunking_related(user_input, chat_history):
        return {"answer": OFF_TOPIC_MESSAGE, "context": []}
    chain = get_rag_chain(infer_coach_filter(user_input))
    response = chain.invoke({"input": user_input, "chat_history": chat_history or []})
    return {
        "answer": response.get("answer"),
        "context": [doc.page_content for doc in response.get("context", [])],
    }
