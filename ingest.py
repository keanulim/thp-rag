import subprocess
import sys
from datetime import datetime

# Runs the full ingestion pipeline in order. Each step is already
# incremental/idempotent on its own (main.py and clean.py skip files that
# already exist, chunk.py skips videos already recorded in
# chunked_files.json and never wipes the index by default) — so running this
# repeatedly on a schedule only does work for genuinely new videos.
STEPS = ["main.py", "clean.py", "chunk.py", "script.py"]


def run_step(step: str):
    print(f"\n{'=' * 60}\n{datetime.now().isoformat()} — running {step}\n{'=' * 60}", flush=True)
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"⚠️ {step} exited with code {result.returncode} — stopping pipeline.", flush=True)
        sys.exit(result.returncode)


if __name__ == "__main__":
    for step in STEPS:
        run_step(step)
    print(f"\n{datetime.now().isoformat()} — ingest pipeline complete.", flush=True)
