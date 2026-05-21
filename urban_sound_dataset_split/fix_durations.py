import librosa
import numpy as np

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)

def load_audio_fixed(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    if len(y) < TARGET_LENGTH:
        pad_amount = TARGET_LENGTH - len(y)
        y = np.pad(y, (0, pad_amount), mode="constant")
    else:
        y = y[:TARGET_LENGTH]

    return y