import os
import json
import time
import random
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

CHANNELS = {
    "THP_Strength": "thpstrength1",
    "John_Evans": "JohnEvans",
}

SKIP_LOG = "scrape_skipped.json"


def get_channel_videos(handle):
    """List a channel's videos via yt-dlp.

    scrapetube parses YouTube's page HTML for a "videoRenderer" key that no
    longer exists there — YouTube migrated channel grids to a "lockupViewModel"
    component, so scrapetube.get_channel() silently returns zero results.
    yt-dlp is actively maintained against exactly this kind of change.
    """
    url = f"https://www.youtube.com/@{handle}/videos"
    ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [
            {"videoId": entry["id"], "title": entry.get("title") or ""}
            for entry in info.get("entries", [])
            if entry and entry.get("id")
        ]


def load_skip_log():
    if os.path.exists(SKIP_LOG):
        with open(SKIP_LOG, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def log_skip(skipped, video_id, title, reason):
    skipped.append({"video_id": video_id, "title": title, "reason": reason})
    with open(SKIP_LOG, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)


def download_transcripts():
    ytt_api = YouTubeTranscriptApi()
    skipped = load_skip_log()

    for folder_name, handle in CHANNELS.items():
        os.makedirs(folder_name, exist_ok=True)
        print(f"\n🚀 Processing Channel: {handle}")

        videos = get_channel_videos(handle)
        print(f"Found {len(videos)} videos")

        for i, video in enumerate(videos):
            video_id = video["videoId"]
            title = video["title"]

            # Keyed by video_id, not title — titles can collide or change,
            # video_id can't.
            file_path = f"{folder_name}/{video_id}.json"

            if os.path.exists(file_path):
                print(f"⏭️ Skipping (already exists): {title}")
                continue

            if i > 0 and i % 10 == 0:
                print("☕ Batch complete. Taking a 5-minute break...")
                time.sleep(300)

            wait_time = random.uniform(10.0, 25.0)
            print(f"Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            try:
                transcript_list = ytt_api.list(video_id)

                # Manually-created transcripts are preferred automatically
                # over auto-generated ones when both exist.
                transcript = transcript_list.find_transcript(["en"])
                fetched_data = transcript.fetch()

                segments = [
                    {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
                    for snippet in fetched_data
                ]

                record = {
                    "video_id": video_id,
                    "title": title,
                    "is_generated": fetched_data.is_generated,
                    "segments": segments,
                }

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)

                kind = "auto-generated" if fetched_data.is_generated else "manual"
                print(f"success ({kind}): {title}")

            except Exception as e:
                reason = str(e)
                if "429" in reason or "RequestBlocked" in reason:
                    print("rate limit, waiting 2 minutes")
                    time.sleep(120)
                    log_skip(skipped, video_id, title, "rate_limited_gave_up_this_pass")
                else:
                    print(f"skipping {video_id}: {reason[:50]}...")
                    log_skip(skipped, video_id, title, reason[:200])


if __name__ == "__main__":
    download_transcripts()
