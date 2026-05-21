from pathlib import Path
import librosa

DATASET_DIR = Path(".")

bad_files = []
duration_warnings = []
total_files = 0

for wav_file in DATASET_DIR.rglob("*"):
    if not wav_file.is_file():
        continue

    if wav_file.suffix.lower() != ".wav":
        continue

    total_files += 1

    try:
        y, sr = librosa.load(wav_file, sr=None, mono=True)
        duration = len(y) / sr

        if duration < 1.8 or duration > 2.2:
            duration_warnings.append((wav_file, duration))

    except Exception as e:
        bad_files.append((wav_file, str(e)))

print(f"Checked {total_files} WAV files.")

if duration_warnings:
    print("\nDuration warnings:")
    for file, duration in duration_warnings:
        print(f"{file} = {duration:.2f}s")
else:
    print("\nNo duration warnings.")

if bad_files:
    print("\nBad files:")
    for file, error in bad_files:
        print(file, error)
else:
    print("\nAll files are readable with librosa.")