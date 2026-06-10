from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .audio import logmel_batch_from_file
from ..settings import CLASSES, MODELS_DIR

_model_cache = {'path': None, 'model': None}


def build_cnn_model() -> keras.Model:
    model = keras.Sequential(name='sequential')
    model.add(layers.Input(shape=(64, 126, 1), name='input_layer'))

    for i, (filters, drop) in enumerate([(32, 0.10), (64, 0.15), (128, 0.20), (256, 0.25)]):
        suffix = '' if i == 0 else f'_{i}'
        model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='linear', name=f'conv2d{suffix}'))
        model.add(layers.BatchNormalization(name=f'batch_normalization{suffix}'))
        model.add(layers.ReLU(name=f're_lu{suffix}'))
        model.add(layers.MaxPooling2D((2, 2), name=f'max_pooling2d{suffix}'))
        model.add(layers.Dropout(drop, name=f'dropout{suffix}'))

    model.add(layers.GlobalAveragePooling2D(name='global_average_pooling2d'))
    model.add(layers.Dense(128, activation='relu', name='dense'))
    model.add(layers.Dropout(0.50, name='dropout_4'))
    model.add(layers.Dense(len(CLASSES), activation='softmax', name='dense_1'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def ensure_initial_model() -> Path:
    version_dir = MODELS_DIR / 'model_versions' / 'uploaded_deeper_cnn'
    version_dir.mkdir(parents=True, exist_ok=True)
    keras_path = version_dir / 'model.keras'
    weights_path = version_dir / 'model.weights.h5'

    if keras_path.exists():
        return keras_path

    model = build_cnn_model()
    if weights_path.exists():
        model.load_weights(str(weights_path))
    model.save(str(keras_path))
    return keras_path


def load_model(model_path: str | Path):
    path = str(model_path)

    if _model_cache['path'] == path and _model_cache['model'] is not None:
        return _model_cache['model']

    p = Path(path)
    if not p.exists():
        p = ensure_initial_model()

    model = keras.models.load_model(str(p), compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    _model_cache['path'] = str(p)
    _model_cache['model'] = model
    return model


def predict_cnn(
    file_path: str | Path,
    model_path: str | Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray:
    """
    Tahmin mantığı:
    - Kullanıcının seçtiği aralığı alır.
    - Bu aralığı 2 saniyelik kayan pencerelere böler.
    - Her pencere için tahmin alır.
    - Son tahmin, pencerelerin ortalamasıdır.
    """
    model = load_model(model_path)
    x = logmel_batch_from_file(file_path, scan=True, start_sec=start_sec, end_sec=end_sec)
    preds = model.predict(x, verbose=0)
    return preds.mean(axis=0)
