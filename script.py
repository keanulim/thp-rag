import scrapetube
import json

CHANNELS = {"THP_Strength": "thpstrength1", "John_Evans": "JohnEvans"}  # Add your channels
links_map = {}

for folder_name, handle in CHANNELS.items():
    print(f"🔗 Generating links for: {handle}")
    videos = scrapetube.get_channel(channel_username=handle)

    for video in videos:
        video_id = video['videoId']
        title = video['title']['runs'][0]['text']

        # This MUST match the safe_title logic from your scraper script
        safe_title = "".join([c for c in title if c.isalnum() or c == ' ']).strip()

        # Map the filename (title) to the full URL
        links_map[safe_title] = f"https://www.youtube.com/watch?v={video_id}"

# Save this to a file that app.py can read
with open("youtube_links.json", "w") as f:
    json.dump(links_map, f, indent=4)

print(f"✅ Created youtube_links.json with {len(links_map)} mappings.")