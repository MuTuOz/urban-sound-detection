from __future__ import annotations
import os
from pathlib import Path

ROOT = Path('/app')
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
UPLOAD_DIR = DATA_DIR / 'uploads'
BASE_DATASET_DIR = DATA_DIR / 'base_dataset'
DB_PATH = DATA_DIR / 'akinci.db'
STATE_PATH = MODELS_DIR / 'model_state.json'
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'akincises-admin')
CLASSES = [
    'bird_sounds', 'construction', 'crowd_chatter', 'footsteps',
    'public_transport', 'siren', 'traffic', 'wind_trees'
]
CLASS_TR = {
    'bird_sounds': 'Kuş sesleri',
    'construction': 'İnşaat',
    'crowd_chatter': 'Kalabalık / konuşma',
    'footsteps': 'Ayak sesleri',
    'public_transport': 'Toplu taşıma',
    'siren': 'Siren',
    'traffic': 'Trafik',
    'wind_trees': 'Rüzgâr / ağaçlar',
}
PROMOTE_MIN_VAL_ACC = float(os.getenv('PROMOTE_MIN_VAL_ACC', '0.50'))
TRAIN_EPOCHS = int(os.getenv('TRAIN_EPOCHS', '10'))
ENSEMBLE_CNN_WEIGHT = float(os.getenv('ENSEMBLE_CNN_WEIGHT', '0.60'))
ENSEMBLE_YAMNET_WEIGHT = float(os.getenv('ENSEMBLE_YAMNET_WEIGHT', '0.40'))
