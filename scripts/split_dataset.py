from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(".")
METADATA_CSV = DATASET_DIR / "clips_metadata.csv"
OUTPUT_DIR = DATASET_DIR.parent / "urban_sound_dataset_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

df = pd.read_csv(METADATA_CSV)

# First split: train vs temp
train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df["label"],
    random_state=RANDOM_STATE
)

# Second split: val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Create folders and copy files
for _, row in split_df.iterrows():
    source_path = DATASET_DIR / row["relative_path"]
    target_dir = OUTPUT_DIR / row["split"] / row["label"]
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / row["clip_filename"]
    shutil.copy2(source_path, target_path)

# Save updated metadata
split_metadata_path = OUTPUT_DIR / "clips_metadata_with_splits.csv"
split_df.to_csv(split_metadata_path, index=False)

print(f"Created split dataset at: {OUTPUT_DIR}")
print()
print("Split counts:")
print(split_df.groupby(["split", "label"]).size())
print()
print(f"Saved metadata to: {split_metadata_path}")