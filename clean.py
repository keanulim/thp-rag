import os
import json
import time
from google import genai
from google.genai import types

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.genai.errors import ClientError


client = genai.Client(api_key="API_KEY")

COACH_FOLDERS = ["John_Evans", "THP_Strength"]

SYSTEM_PROMPT = """
You are an expert Exercise Scientist and Data Engineer specializing in Vertical Jump Mechanics (specifically THP Strength and John Evans methodologies). 
Your goal is to transform messy, auto-generated transcripts into a structured, high-fidelity JSON dataset.

### EXTRACTION RULES:
1. CLEANING: Remove all 'narrative fluff'. 
2. PUNCTUATION: Fix run-on sentences. 
3. TERMINOLOGY: Capitalize 'Plyometrics', 'Isometrics', and 'RFD'.
4. NUMERIC DATA: Extract exact numbers for Sets, Reps, and Intensity (RPE/RIR).
5. METADATA: Every field must be populated according to the schema below.

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

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ClientError)
)
def generate_refined_content(raw_text):
    return client.models.generate_content(
        model='gemini-2.0-flash',
        contents=raw_text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type='application/json',
        )
    )


def clean_and_tag_files():
    for folder in COACH_FOLDERS:
        output_folder = f"Cleaned_{folder}"
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]

        for filename in files:
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.txt', '.json'))

            if os.path.exists(output_path):
                continue

            with open(input_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            try:
                response = generate_refined_content(raw_text)

                if not response or not response.text:
                    continue

                clean_json_str = response.text.strip().replace('```json', '').replace('```', '')
                parsed_json = json.loads(clean_json_str)

                entries = [parsed_json] if isinstance(parsed_json, dict) else parsed_json

                cleaned_entries = []
                for entry in entries:
                    if 'metadata' in entry:
                        entry.update(entry.pop('metadata'))

                    if 'quantitative_stats' in entry:
                        entry['stats'] = entry.pop('quantitative_stats')

                    entry['coach'] = folder.replace('_', ' ')
                    entry['source_filename'] = filename
                    cleaned_entries.append(entry)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_entries, f, indent=4)

                print(f"cleaned: {filename}")
                time.sleep(2)  

            except Exception as e:
                print(f"error processing {filename}: {e}")


if __name__ == "__main__":
    clean_and_tag_files()
