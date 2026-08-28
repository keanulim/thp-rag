import scrapetube
import os
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi

# Updated channel mapping for 2026
CHANNELS = {
    "THP_Strength": "thpstrength1"
}


def download_transcripts():
    # Initialize the new instance-based API
    ytt_api = YouTubeTranscriptApi()

    for folder_name, handle in CHANNELS.items():
        os.makedirs(folder_name, exist_ok=True)
        print(f"\n🚀 Processing Channel: {handle}")

        # Get all videos (using handle/username)
        videos = scrapetube.get_channel(channel_username=handle)

        for i, video in enumerate(videos):
            video_id = video['videoId']
            title = video['title']['runs'][0]['text']

            safe_title = "".join([c for c in title if c.isalnum() or c == ' ']).strip()
            file_path = f"{folder_name}/{safe_title}.txt"

            if os.path.exists(file_path):
                print(f"⏭️ Skipping (already exists): {title}")
                continue

                # NEW: Long break every 10 videos
            if i > 0 and i % 10 == 0:
                print("☕ Batch complete. Taking a 1-minute coffee break...")
                time.sleep(300)

                    # NEW: Increased random jitter
            wait_time = random.uniform(10.0, 25.0)  # Much slower
            print(f"Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            try:
                # 2. DISCOVERY: List all transcripts (including auto-generated)
                transcript_list = ytt_api.list(video_id)

                # 3. SELECTION: Prefer manual English, fallback to auto 'en'
                transcript = transcript_list.find_transcript(['en'])

                # 4. EXTRACTION: Access '.text' attribute of FetchedTranscriptSnippet objects
                fetched_data = transcript.fetch()
                full_text = " ".join([snippet.text for snippet in fetched_data])

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"success: {title}")

            except Exception as e:
                # Catch specific rate limit errors
                if "429" in str(e) or "RequestBlocked" in str(e):
                    print("rate limit, waiting 2 minutes")
                    time.sleep(120)
                else:
                    print(f"skipping {video_id}: {str(e)[:50]}...")


if __name__ == "__main__":
    download_transcripts()