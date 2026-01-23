import scrapetube
import os
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi

CHANNELS = {
    "THP_Strength": "thpstrength1"
}


def download_transcripts():
    ytt_api = YouTubeTranscriptApi()

    for folder_name, handle in CHANNELS.items():
        os.makedirs(folder_name, exist_ok=True)
        print(f"\n processing: {handle}")

        videos = scrapetube.get_channel(channel_username=handle)

        for i, video in enumerate(videos):
            video_id = video['videoId']
            title = video['title']['runs'][0]['text']

            safe_title = "".join([c for c in title if c.isalnum() or c == ' ']).strip()
            file_path = f"{folder_name}/{safe_title}.txt"

            if os.path.exists(file_path):
                print(f" skipping (already exists): {title}")
                continue

                # avoid api limits
            if i > 0 and i % 10 == 0:
                print("buffering...")
                time.sleep(300)

                    #random jitter
            wait_time = random.uniform(10.0, 25.0)  
            print(f"Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            try:
                transcript_list = ytt_api.list(video_id)
                transcript = transcript_list.find_transcript(['en'])

                fetched_data = transcript.fetch()
                full_text = " ".join([snippet.text for snippet in fetched_data])

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                print(f"success: {title}")

            except Exception as e:
                if "429" in str(e) or "RequestBlocked" in str(e):
                    print("rate limit, waiting 2 minutes")
                    time.sleep(120)
                else:
                    print(f"skipping {video_id}: {str(e)[:50]}...")


if __name__ == "__main__":
    download_transcripts()
