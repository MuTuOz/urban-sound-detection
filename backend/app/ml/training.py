from __future__ import annotations

from pathlib import Path
import json
import shutil
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import joblib
import tensorflow as tf
from tensorflow import keras

from .audio import (
    AUDIO_EXTS,
    logmel,
    load_audio,
    windows,
    save_audio_range,
    embedding_from_file,
)
from .modeling import build_cnn_model
from ..settings import CLASSES, BASE_DATASET_DIR, MODELS_DIR, PROMOTE_MIN_VAL_ACC, TRAIN_EPOCHS
from ..state import read_state, update_state
from ..db import connect, row_to_dict

# Güvenlik için ayardaki eşik düşük kalsa bile 0.70 altını aktif etmiyoruz.
SAFE_MIN_PROMOTE_VAL_ACC = max(float(PROMOTE_MIN_VAL_ACC), 0.70)

# Yeni model eski modelden çok az düşükse rastgele split farkı olabilir diye küçük tolerans.
OLD_MODEL_TOLERANCE = 0.005


def list_base_files():
    items = []

    for idx, cls in enumerate(CLASSES):
        d = BASE_DATASET_DIR / cls
        d.mkdir(parents=True, exist_ok=True)

        for p in d.rglob('*'):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                items.append((p, idx))

    return items


def _as_float_or_none(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except Exception:
        return None


def copy_feedback_to_base(record_ids: list[str] | None = None) -> int:
    """
    Admin onaylı kayıtları base_dataset içine kopyalar.

    v12 farkı:
    Kullanıcı bir ses aralığı seçtiyse, bütün dosyayı kopyalamak yerine sadece o aralığı
    WAV olarak base_dataset'e yazar. Böylece retraining doğru ses bölümünden öğrenir.
    """
    conn = connect()

    if record_ids:
        q = f"SELECT * FROM uploads WHERE id IN ({','.join(['?'] * len(record_ids))})"
        rows = conn.execute(q, record_ids).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM uploads WHERE admin_label IS NOT NULL AND admin_label != ''"
        ).fetchall()

    copied = 0

    for row in rows:
        r = row_to_dict(row)
        label = r.get('admin_label') or r.get('user_label')

        if label not in CLASSES:
            continue

        src = Path(r['stored_path'])
        if not src.exists():
            src = Path('/app') / r['stored_path'].lstrip('/')
        if not src.exists():
            continue

        dst_dir = BASE_DATASET_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        start_sec = _as_float_or_none(r.get('selected_start_sec'))
        end_sec = _as_float_or_none(r.get('selected_end_sec'))

        # Seçili aralık varsa eğitim datasına sadece o bölüm girsin.
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            dst = dst_dir / f"feedback_{r['id']}_{start_sec:.1f}_{end_sec:.1f}.wav"
            if not dst.exists():
                save_audio_range(src, dst, start_sec=start_sec, end_sec=end_sec)
                copied += 1
        else:
            dst = dst_dir / f"feedback_{r['id']}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

        conn.execute(
            'UPDATE uploads SET status=? WHERE id=?',
            ('approved', r['id']),
        )

    conn.commit()
    conn.close()

    return copied


def load_dataset_arrays(items):
    """
    Dataset dosyalarını modele uygun X/y dizilerine çevirir.

    v12 farkı:
    Dosya 2 saniyeden uzunsa tek ilk parçayı almak yerine 2 saniyelik kayan
    pencereler üretir. Örneğin 5 saniyelik seçili aralık için yaklaşık:
    0-2, 1-3, 2-4, 3-5 sn parçaları kullanılır.
    """
    X, y = [], []

    for path, idx in items:
        try:
            audio = load_audio(path)
            for seg in windows(audio, hop_seconds=1.0, max_windows=12):
                X.append(logmel(seg))
                y.append(idx)
        except Exception as e:
            print(f'[SKIP] {path}: {e}', flush=True)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def get_model_val_accuracy(version: str | None) -> float | None:
    if not version:
        return None

    metrics_path = MODELS_DIR / 'model_versions' / version / 'metrics.json'
    if not metrics_path.exists():
        return None

    try:
        data = json.loads(metrics_path.read_text(encoding='utf-8'))
        val = data.get('val_accuracy')
        return float(val) if val is not None else None
    except Exception as e:
        print(f'[METRICS READ SKIP] {metrics_path}: {e}', flush=True)
        return None


def should_promote_cnn_model(
    new_val_acc: float,
    old_version: str | None,
) -> tuple[bool, list[str], float | None]:
    reasons: list[str] = []
    old_val_acc = get_model_val_accuracy(old_version)

    if new_val_acc < SAFE_MIN_PROMOTE_VAL_ACC:
        reasons.append(
            f'validation skoru düşük: {new_val_acc:.4f} < minimum {SAFE_MIN_PROMOTE_VAL_ACC:.2f}'
        )

    if old_val_acc is not None:
        if new_val_acc < (old_val_acc - OLD_MODEL_TOLERANCE):
            reasons.append(
                f'yeni model eski aktif modelden daha iyi değil: yeni={new_val_acc:.4f}, eski={old_val_acc:.4f}'
            )

    return len(reasons) == 0, reasons, old_val_acc


def train_cnn_job(params: dict) -> dict:
    ids = params.get('record_ids') or None

    copied = copy_feedback_to_base(
        ids if params.get('mode') == 'selected' else None
    )

    update_state(
        training_state='running',
        training_message=(
            'CNN retraining çalışıyor. Base dataset + onaylı kayıtlar kullanılıyor. '
            'Seçili aralıklar varsa sadece o aralıklar eğitim verisine eklenir. '
            'Eski aktif model tahmin vermeye devam eder.'
        ),
    )

    state_before_training = read_state()
    old_active_version = state_before_training.get('active_version')
    old_active_model_path = state_before_training.get('active_model_path')

    items = list_base_files()

    if len(items) < len(CLASSES) * 2:
        msg = 'Dataset yetersiz. Her sınıfta en az birkaç ses dosyası olmalı.'
        update_state(training_state='failed', training_message=msg)
        return {'ok': False, 'message': msg}

    item_labels = np.asarray([idx for _, idx in items], dtype=np.int64)

    if len(set(item_labels.tolist())) < len(CLASSES):
        missing = [
            CLASSES[i]
            for i in range(len(CLASSES))
            if i not in set(item_labels.tolist())
        ]
        msg = f'Eksik sınıflar var: {", ".join(missing)}'
        update_state(training_state='failed', training_message=msg)
        return {'ok': False, 'message': msg}

    # Split'i dosya seviyesinde yapıyoruz. Böylece aynı 5 saniyelik kayıttan çıkan
    # 2 saniyelik pencereler hem train hem validation'a karışmaz.
    class_counts = np.bincount(item_labels, minlength=len(CLASSES))
    stratify = item_labels if min(class_counts) >= 2 else None

    train_items, val_items = train_test_split(
        items,
        test_size=0.18,
        random_state=42,
        stratify=stratify,
    )

    X_train, y_train = load_dataset_arrays(train_items)
    X_val, y_val = load_dataset_arrays(val_items)

    if len(X_train) == 0 or len(X_val) == 0:
        msg = 'Eğitim/validation verisi üretilemedi. Ses dosyalarını kontrol et.'
        update_state(training_state='failed', training_message=msg)
        return {'ok': False, 'message': msg}

    model = build_cnn_model()

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(CLASSES)),
        y=y_train,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=32,
        verbose=2,
        class_weight=dict(enumerate(weights)),
        callbacks=callbacks,
    )

    loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    version = 'retrained_' + time.strftime('%Y%m%d_%H%M%S')
    vdir = MODELS_DIR / 'model_versions' / version
    vdir.mkdir(parents=True, exist_ok=True)

    model_path = vdir / 'model.keras'
    model.save(str(model_path))

    metrics = {
        'val_accuracy': float(val_acc),
        'val_loss': float(loss),
        'files_total': int(len(items)),
        'files_train': int(len(train_items)),
        'files_val': int(len(val_items)),
        'segments_train': int(len(X_train)),
        'segments_val': int(len(X_val)),
        'feedback_copied': int(copied),
        'safe_min_promote_val_acc': float(SAFE_MIN_PROMOTE_VAL_ACC),
        'old_active_version': old_active_version,
    }

    (vdir / 'metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    promote, reasons, old_val_acc = should_promote_cnn_model(
        new_val_acc=float(val_acc),
        old_version=old_active_version,
    )

    if promote:
        update_state(
            active_version=version,
            active_model_path=str(model_path),
            training_state='finished',
            training_message=(
                f'CNN retraining tamamlandı. Aktif model {version} oldu. '
                f'Val accuracy={val_acc:.4f}.'
            ),
            last_finished_version=version,
        )
        ok = True
    else:
        reason_text = '; '.join(reasons) if reasons else 'model aktif etmek için yeterli görülmedi'

        update_state(
            training_state='finished_not_promoted',
            training_message=(
                f'CNN retraining tamamlandı ama yeni model aktif edilmedi. '
                f'Sebep: {reason_text}. '
                f'Eski aktif model kullanılmaya devam ediyor: {old_active_version or "-"}. '
                f'Candidate: {version}.'
            ),
            last_finished_version=version,
        )
        ok = False

    return {
        'ok': ok,
        'version': version,
        'old_active_version': old_active_version,
        'old_active_model_path': old_active_model_path,
        'old_val_accuracy': old_val_acc,
        **metrics,
    }


def train_yamnet_like_job(params: dict) -> dict:
    update_state(
        yamnet_state='running',
        yamnet_message=(
            'YAMNet transfer eğitimi çalışıyor. '
            'Eski aktif model tahmin vermeye devam eder.'
        ),
    )

    items = list_base_files()

    X, y = [], []
    for path, idx in items:
        try:
            X.append(embedding_from_file(path))
            y.append(idx)
        except Exception as e:
            print(f'[YAMNET-LIKE SKIP] {path}: {e}', flush=True)

    if len(X) < len(CLASSES) * 2 or len(set(y)) < len(CLASSES):
        msg = 'YAMNet/embedding eğitimi için dataset yetersiz.'
        update_state(yamnet_state='failed', yamnet_message=msg)
        return {'ok': False, 'message': msg}

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    class_counts = np.bincount(y, minlength=len(CLASSES))
    stratify = y if min(class_counts) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.18,
        random_state=42,
        stratify=stratify,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            n_jobs=1,
        ),
    )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    version = 'yamnet_' + time.strftime('%Y%m%d_%H%M%S')

    ydir = MODELS_DIR / 'yamnet'
    ydir.mkdir(parents=True, exist_ok=True)

    path = ydir / f'{version}.joblib'
    joblib.dump(clf, path)

    meta = {
        'version': version,
        'path': str(path),
        'test_accuracy': float(acc),
        'samples': int(len(X)),
        'note': 'Local embedding classifier used as robust YAMNet transfer fallback.',
    }

    (ydir / f'{version}.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    update_state(
        yamnet_version=version,
        yamnet_state='finished',
        yamnet_message=(
            f'YAMNet transfer eğitimi tamamlandı. '
            f'Test accuracy={acc:.4f}.'
        ),
        yamnet_model_path=str(path),
    )

    return {'ok': True, **meta}


def predict_yamnet_like(
    file_path: str | Path,
    model_path: str | Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
):
    clf = joblib.load(model_path)
    feat = embedding_from_file(file_path, start_sec=start_sec, end_sec=end_sec).reshape(1, -1)
    return clf.predict_proba(feat)[0]
