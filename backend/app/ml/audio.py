from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

SR = 16000
DURATION = 2.0
SAMPLES = int(SR * DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
TARGET_FRAMES = 126
MAX_WINDOWS = 30

AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm'}


def load_audio(path: str | Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if y is None or len(y) == 0:
        y = np.zeros(SAMPLES, dtype=np.float32)
    return y.astype(np.float32)


def audio_duration_seconds(path: str | Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        y = load_audio(path)
        return float(len(y) / SR)


def pad_or_crop(y: np.ndarray, start: int = 0) -> np.ndarray:
    start = max(0, int(start))
    seg = y[start:start + SAMPLES]
    if len(seg) < SAMPLES:
        seg = np.pad(seg, (0, SAMPLES - len(seg)))
    return seg.astype(np.float32)


def crop_audio_range(y: np.ndarray, start_sec: float | None = None, end_sec: float | None = None) -> np.ndarray:
    if start_sec is None:
        start_sec = 0.0
    start_sec = max(0.0, float(start_sec))

    if end_sec is None or float(end_sec) <= start_sec:
        end_sec = len(y) / SR
    end_sec = min(float(end_sec), len(y) / SR)

    start = max(0, int(round(start_sec * SR)))
    end = max(start, int(round(end_sec * SR)))
    seg = y[start:end]

    if len(seg) == 0:
        seg = y[:SAMPLES]

    return seg.astype(np.float32)


def save_audio_range(src: str | Path, dst: str | Path, start_sec: float | None = None, end_sec: float | None = None) -> Path:
    y = load_audio(src)
    seg = crop_audio_range(y, start_sec, end_sec)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), seg, SR)
    return dst


def windows(y: np.ndarray, hop_seconds: float = 1.0, max_windows: int = MAX_WINDOWS) -> list[np.ndarray]:
    if len(y) <= SAMPLES:
        return [pad_or_crop(y, 0)]

    hop = max(1, int(SR * hop_seconds))
    last_start = max(0, len(y) - SAMPLES)
    starts = list(range(0, last_start + 1, hop))

    if not starts:
        starts = [0]

    if starts[-1] != last_start:
        starts.append(last_start)

    return [pad_or_crop(y, s) for s in starts[:max_windows]]


def logmel(segment: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)

    if db.shape[1] < TARGET_FRAMES:
        db = np.pad(
            db,
            ((0, 0), (0, TARGET_FRAMES - db.shape[1])),
            mode='constant',
            constant_values=db.min(),
        )
    elif db.shape[1] > TARGET_FRAMES:
        db = db[:, :TARGET_FRAMES]

    mean = float(db.mean())
    std = float(db.std() + 1e-6)
    x = (db - mean) / std
    return x.astype(np.float32)[..., None]


def logmel_batch_from_file(
    path: str | Path,
    scan: bool = True,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray:
    y = load_audio(path)
    y = crop_audio_range(y, start_sec, end_sec)
    segs = windows(y) if scan else [pad_or_crop(y, 0)]
    return np.stack([logmel(seg) for seg in segs], axis=0)


def embedding_from_file(
    path: str | Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray:
    y = load_audio(path)
    y = crop_audio_range(y, start_sec, end_sec)
    segs = windows(y, hop_seconds=1.0, max_windows=10)

    feats = []
    for seg in segs:
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=40, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feat = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            mel_db.mean(axis=1), mel_db.std(axis=1),
        ]).astype(np.float32)
        feats.append(feat)

    return np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
