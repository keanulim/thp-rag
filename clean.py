import os
import json
import time
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- NEW IMPORTS REQUIRED FOR RETRIES ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.genai.errors import ClientError

# ---------------------------------------

# Without an explicit timeout, a dropped/stalled connection (e.g. after the
# machine wakes from sleep, or a network blip) hangs the request forever —
# no exception is ever raised, so nothing here retries and the whole
# pipeline just sits stuck indefinitely. 120s is generous for a single
# cleaning call but still bounds the worst case.
REQUEST_TIMEOUT_MS = 120_000

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

COACH_FOLDERS = ["John_Evans", "THP_Strength"]

SYSTEM_PROMPT = """
You are an expert Exercise Scientist and Data Engineer specializing in Vertical Jump Mechanics (specifically THP Strength and John Evans methodologies).
Your goal is to lightly format messy, auto-generated transcripts and extract structured metadata from them — WITHOUT rewriting or paraphrasing the actual words.

### EXTRACTION RULES:
1. PUNCTUATION ONLY, NOT REWRITING: Add punctuation, capitalization, and paragraph breaks so the
   transcript reads naturally. Do NOT remove, reorder, summarize, or paraphrase any content —
   preserve the speaker's exact original wording, including tangents and filler. Only insert
   punctuation marks, capitalize sentence starts/proper nouns, and break into paragraphs at natural
   topic shifts. The output should be recognizable as the same words, just formatted.
2. TERMINOLOGY: Capitalize 'Plyometrics', 'Isometrics', and 'RFD'.
3. NUMERIC DATA: Extract exact numbers for Sets, Reps, and Intensity (RPE/RIR) into metadata (do not
   alter numbers in the text itself).
4. METADATA: Every field must be populated according to the schema below.
5. SPEAKER ATTRIBUTION: These transcripts sometimes feature a guest or athlete speaking in first
   person about their own training or results, not the channel's coach. Read the full transcript
   and insert an inline tag "[SPEAKER: <name or role>]" immediately before each stretch of text
   spoken by a different person than the one before it (e.g. use cues like self-introductions,
   being addressed by name, or a host asking someone else about "your" results).
   - Recurring guests/athletes to recognize by name if mentioned or self-introduced: Josh Ruble,
     Dom Gonzales (also goes by "Dom Dunks"), Donovan Hawkins, Austin, Ben Moxness. If a stretch is
     clearly one of these people, tag them by their name above (use "Dom Gonzales" as the canonical
     name even if the transcript says "Dom Dunks").
   - Do not tag every sentence — only tag at genuine speaker changes.
   - If you cannot confidently tell who is speaking a given stretch, use "[SPEAKER: Uncertain]"
     rather than guessing a name.
   - If the whole transcript is clearly one person speaking throughout, one tag at the start is enough.

### SCHEMA:
{
  "cleaned_text": "text",       
  "metadata": {
    "primary_focus": "focus",
    "exercise_list": [],
    "quantitative_stats": {"reps_per_set": null, "total_sets": null, "intensity_rpe": null},
    "movement_taxonomy": {"jump_type": "type", "plant_foot": "foot", "body_part": []},
    "difficulty": "Intermediate",
    "is_injury_prevention": false
  }
}

### OUTPUT FORMAT:
Return ONLY the raw JSON object.
"""


# The decorator MUST be immediately above the function with no empty lines
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ClientError, httpx.TimeoutException))
)
def generate_refined_content(raw_text):
    return client.models.generate_content(
        model='gemini-3.7-flash',
        contents=raw_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    )


def clean_and_tag_files():
    for folder in COACH_FOLDERS:
        output_folder = f"Cleaned_{folder}"
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(folder) if f.endswith('.json')]

        for filename in files:
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, filename)

            if os.path.exists(output_path):
                continue

            with open(input_path, 'r', encoding='utf-8') as f:
                scraped = json.load(f)

            raw_text = " ".join(segment["text"] for segment in scraped["segments"])

            try:
                # Use the retry-protected function
                response = generate_refined_content(raw_text)

                if not response or not response.text:
                    continue

                # Sanitize response string (remove potential markdown junk)
                clean_json_str = response.text.strip().replace('```json', '').replace('```', '')
                parsed_json = json.loads(clean_json_str)

                # Normalize to list
                entries = [parsed_json] if isinstance(parsed_json, dict) else parsed_json

                cleaned_entries = []
                for entry in entries:
                    # Logic to flatten and clean
                    if 'metadata' in entry:
                        entry.update(entry.pop('metadata'))

                    if 'quantitative_stats' in entry:
                        entry['stats'] = entry.pop('quantitative_stats')

                    entry['coach'] = folder.replace('_', ' ')
                    entry['video_id'] = scraped["video_id"]
                    entry['video_title'] = scraped["title"]
                    entry['is_generated_transcript'] = scraped["is_generated"]
                    # Carried through so chunk.py can map cleaned/chunked text
                    # back to an approximate original timestamp.
                    entry['segments'] = scraped["segments"]
                    cleaned_entries.append(entry)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_entries, f, indent=4)

                print(f"✨ Refined: {filename}")
                time.sleep(2)  # Small buffer

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")


if __name__ == "__main__":
    clean_and_tag_files()