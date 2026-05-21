from pathlib import Path
import pandas as pd
import wave

DATASET_DIR = Path("processed_clips")   # change this if needed
OUTPUT_CSV = Path("clips_metadata.csv")

rows = []

def get_wav_duration(file_path):
    try:
        with wave.open(str(file_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except Exception:
        return None

for class_dir in DATASET_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    label = class_dir.name

    for wav_file in class_dir.glob("*.wav"):
        duration = get_wav_duration(wav_file)

        rows.append({
            "clip_filename": wav_file.name,
            "relative_path": str(wav_file.relative_to(DATASET_DIR)),
            "label": label,
            "duration_sec": round(duration, 3) if duration else "",
            "device": "Zoom H4essential",
            "split": ""
        })

df = pd.DataFrame(rows)
df = df.sort_values(["label", "clip_filename"])

df.to_csv(OUTPUT_CSV, index=False)

print(f"Created {OUTPUT_CSV} with {len(df)} clips.")
print(df["label"].value_counts())