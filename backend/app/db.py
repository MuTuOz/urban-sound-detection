from __future__ import annotations

import sqlite3
import json
import time
from .settings import DATA_DIR, DB_PATH

SCHEMA = '''
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_prediction TEXT,
    probabilities_json TEXT,
    model_version TEXT,
    user_label TEXT,
    admin_label TEXT,
    status TEXT DEFAULT 'pending',
    selected_start_sec REAL,
    selected_end_sec REAL,
    duration_sec REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    state TEXT NOT NULL,
    message TEXT,
    params_json TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
'''

UPLOAD_EXTRA_COLUMNS = {
    'selected_start_sec': 'REAL',
    'selected_end_sec': 'REAL',
    'duration_sec': 'REAL',
}


def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        rows = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
        return {row['name'] for row in rows}
    except Exception:
        return set()


def _ensure_upload_columns(conn: sqlite3.Connection):
    """
    Eski database dosyalarında yeni kolonlar olmayabilir.
    Bu fonksiyon eksik kolonları ekler.

    Not: Bazı SQLite sürümlerinde / eski migration durumlarında PRAGMA kontrolü
    ile ALTER TABLE arası çakışma yaşanabiliyor. Bu yüzden duplicate column
    hatasını güvenli şekilde yok sayıyoruz.
    """

    existing = _table_columns(conn, 'uploads')

    for name, typ in UPLOAD_EXTRA_COLUMNS.items():
        if name in existing:
            continue

        try:
            conn.execute(f'ALTER TABLE uploads ADD COLUMN {name} {typ}')
            existing.add(name)
        except sqlite3.OperationalError as e:
            # Kolon zaten varsa uygulama açılışını bozma.
            if 'duplicate column name' in str(e).lower():
                existing.add(name)
                continue
            raise


def init_db():
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        _ensure_upload_columns(conn)
        conn.commit()
    finally:
        conn.close()


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def row_to_dict(row):
    d = dict(row)

    if 'probabilities_json' in d and d['probabilities_json']:
        try:
            d['probabilities'] = json.loads(d['probabilities_json'])
        except Exception:
            d['probabilities'] = {}

    if 'params_json' in d and d['params_json']:
        try:
            d['params'] = json.loads(d['params_json'])
        except Exception:
            d['params'] = {}

    if 'result_json' in d and d['result_json']:
        try:
            d['result'] = json.loads(d['result_json'])
        except Exception:
            d['result'] = {}

    return d
