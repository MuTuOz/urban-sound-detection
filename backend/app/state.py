from __future__ import annotations
import json
import time
from pathlib import Path
from .settings import MODELS_DIR, STATE_PATH

DEFAULT_STATE = {
    'active_version': 'uploaded_deeper_cnn',
    'active_model_path': '/app/models/model_versions/uploaded_deeper_cnn/model.keras',
    'training_state': 'idle',
    'training_message': 'Sistem hazır.',
    'last_finished_version': None,
    'yamnet_version': None,
    'yamnet_state': 'idle',
    'yamnet_message': 'YAMNet/embedding modeli henüz eğitilmedi.',
    'updated_at': None,
}

def read_state() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        write_state(DEFAULT_STATE.copy())
    try:
        data = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    except Exception:
        data = DEFAULT_STATE.copy()
    merged = DEFAULT_STATE.copy()
    merged.update(data)
    return merged

def write_state(data: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    tmp = STATE_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATE_PATH)

def update_state(**kwargs) -> dict:
    data = read_state()
    data.update(kwargs)
    write_state(data)
    return data
