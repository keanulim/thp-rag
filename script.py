import json
import yt_dlp

CHANNELS = {"THP_Strength": "thpstrength1", "John_Evans": "JohnEvans"}  # Add your channels
links_map = {}


def get_channel_videos(handle):
    url = f"https://www.youtube.com/@{handle}/videos"
    ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return [
            {"videoId": entry["id"]}
            for entry in info.get("entries", [])
            if entry and entry.get("id")
        ]


for folder_name, handle in CHANNELS.items():
    print(f"🔗 Generating links for: {handle}")
    videos = get_channel_videos(handle)

    for video in videos:
        video_id = video['videoId']

        # Keyed by video_id — stable across title edits/collisions, and
        # matches the video_id stored in each chunk's Pinecone metadata.
        links_map[video_id] = f"https://www.youtube.com/watch?v={video_id}"

# Save this to a file that app.py can read
with open("youtube_links.json", "w") as f:
    json.dump(links_map, f, indent=4)

print(f"✅ Created youtube_links.json with {len(links_map)} mappings.")
