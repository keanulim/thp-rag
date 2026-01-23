Vetted Vertical
A RAG-based search tool for vertical jump training. It uses Google Gemini and Pinecone to answer technical training questions based on transcripts from THP Strength and John Evans.

How it works
Scrape: Pulls transcripts from YouTube.

Clean: Uses LLMs to turn messy transcripts into structured training data.

Search: Uses vector embeddings to find the most relevant video clips.

Answer: Gemini synthesizes the clips into a coaching response with sources.

Tech Stack
Language: Python

AI: Gemini 3 Flash + Text Embeddings

Database: Pinecone (Vector Store)

UI: Streamlit

Setup
Add your GOOGLE_API_KEY and PINECONE_API_KEY to your environment.

Install requirements: uv pip install -r requirements.txt

Run: streamlit run app.py

Keanu Lim UC Berkeley
