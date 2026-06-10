from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from .settings import UPLOAD_DIR, DATA_DIR, BASE_DATASET_DIR, MODELS_DIR, CLASSES, CLASS_TR, ADMIN_TOKEN
from .db import init_db, connect, now, row_to_dict
from .state import read_state, update_state
from .ml.audio import audio_duration_seconds, save_audio_range
from .ml.modeling import ensure_initial_model, predict_cnn
from .ml.training import predict_yamnet_like

app = FastAPI(title='Akıncı Ses', default_response_class=JSONResponse)

FRONTEND = Path('/app/frontend')
app.mount('/static', StaticFiles(directory=str(FRONTEND)), name='static')
app.mount('/assets', StaticFiles(directory=str(FRONTEND)), name='assets')


class FeedbackIn(BaseModel):
    upload_id: str
    user_label: str


class AdminLabelIn(BaseModel):
    admin_label: str


class RetrainIn(BaseModel):
    mode: str = 'all'
    record_ids: list[str] = []


def check_admin(token: str | None):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail='Admin token hatalı.')


def probs_to_payload(probs):
    order = np.argsort(probs)[::-1]
    return [
        {
            'label': CLASSES[i],
            'label_tr': CLASS_TR[CLASSES[i]],
            'probability': float(probs[i]),
            'percent': round(float(probs[i]) * 100, 1),
        }
        for i in order
    ]


def normalize_range(duration: float, start_sec: float | None, end_sec: float | None) -> tuple[float, float]:
    if start_sec is None:
        start_sec = 0.0
    start_sec = max(0.0, float(start_sec))

    if end_sec is None or float(end_sec) <= start_sec:
        if duration > 0:
            end_sec = min(duration, start_sec + 5.0)
        else:
            end_sec = start_sec + 5.0
    else:
        end_sec = float(end_sec)

    if duration > 0:
        start_sec = min(start_sec, max(0.0, duration - 0.1))
        end_sec = min(end_sec, duration)

    if end_sec <= start_sec:
        end_sec = start_sec + 2.0

    return float(start_sec), float(end_sec)


@app.on_event('startup')
def startup():
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (BASE_DATASET_DIR / c).mkdir(parents=True, exist_ok=True)
    p = ensure_initial_model()
    st = read_state()
    if not Path(st.get('active_model_path', '')).exists():
        update_state(
            active_version='uploaded_deeper_cnn',
            active_model_path=str(p),
            training_state='idle',
            training_message='Sistem hazır.',
        )


@app.get('/')
def index():
    return FileResponse(FRONTEND / 'index.html', media_type='text/html; charset=utf-8')


@app.get('/admin')
def admin_page():
    return FileResponse(FRONTEND / 'admin.html', media_type='text/html; charset=utf-8')


@app.get('/admin.html')
def admin_html():
    return admin_page()


@app.get('/favicon.ico')
def favicon():
    return FileResponse(FRONTEND / 'favicon.ico')


@app.get('/api/classes')
def classes():
    return [{'label': c, 'label_tr': CLASS_TR[c]} for c in CLASSES]


@app.get('/api/model/status')
def model_status():
    return read_state()


@app.post('/api/predict')
async def predict(
    file: UploadFile = File(...),
    start_sec: float | None = Form(None),
    end_sec: float | None = Form(None),
):
    if not file.filename:
        raise HTTPException(400, 'Dosya seçilmedi.')

    upload_id = str(uuid.uuid4())
    safe_name = ''.join(ch if ch.isalnum() or ch in '._- ' else '_' for ch in file.filename)[:120]
    if not Path(safe_name).suffix:
        safe_name += '.wav'

    stored = UPLOAD_DIR / f'{upload_id}_{safe_name}'
    with stored.open('wb') as f:
        shutil.copyfileobj(file.file, f)

    duration = audio_duration_seconds(stored)
    selected_start, selected_end = normalize_range(duration, start_sec, end_sec)

    st = read_state()
    cnn_probs = predict_cnn(
        stored,
        st['active_model_path'],
        start_sec=selected_start,
        end_sec=selected_end,
    )

    final_probs = cnn_probs
    decision_mode = 'CNN'

    ypath = st.get('yamnet_model_path')
    if ypath and Path(ypath).exists():
        try:
            yam_probs = predict_yamnet_like(
                stored,
                ypath,
                start_sec=selected_start,
                end_sec=selected_end,
            )
            final_probs = 0.60 * cnn_probs + 0.40 * yam_probs
            decision_mode = 'CNN + YAMNet transfer'
        except Exception as e:
            print(f'[YAMNET PREDICT SKIP] {e}', flush=True)
            decision_mode = 'CNN'

    final_probs = final_probs / max(final_probs.sum(), 1e-8)
    payload = probs_to_payload(final_probs)
    top = payload[0]

    conn = connect()
    conn.execute(
        '''
        INSERT INTO uploads(
            id, filename, stored_path, created_at,
            model_prediction, probabilities_json, model_version,
            status, selected_start_sec, selected_end_sec, duration_sec
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ''',
        (
            upload_id,
            file.filename,
            str(stored),
            now(),
            top['label'],
            json.dumps(payload, ensure_ascii=False),
            st.get('active_version'),
            'pending',
            selected_start,
            selected_end,
            duration,
        ),
    )
    conn.commit()
    conn.close()

    return {
        'upload_id': upload_id,
        'filename': file.filename,
        'prediction': top,
        'probabilities': payload,
        'model_version': st.get('active_version'),
        'decision_mode': decision_mode,
        'duration_sec': duration,
        'selected_range': {
            'start_sec': selected_start,
            'end_sec': selected_end,
        },
    }


@app.post('/api/feedback')
def feedback(inp: FeedbackIn):
    if inp.user_label not in CLASSES:
        raise HTTPException(400, 'Geçersiz sınıf.')

    conn = connect()
    cur = conn.execute(
        'UPDATE uploads SET user_label=?, status=? WHERE id=?',
        (inp.user_label, 'waiting_admin', inp.upload_id),
    )
    conn.commit()
    conn.close()

    if cur.rowcount == 0:
        raise HTTPException(404, 'Kayıt bulunamadı.')

    update_state(
        training_state='waiting_for_admin',
        training_message='Yeni kullanıcı cevabı kaydedildi. Admin panelinden kontrol edilebilir.',
    )
    return {
        'ok': True,
        'message': 'Geri bildirimin kaydedildi. Admin onayından sonra model güncellemesinde kullanılacak.',
    }


@app.get('/api/admin/records')
def admin_records(token: str = Query(...)):
    check_admin(token)
    conn = connect()
    rows = conn.execute('SELECT * FROM uploads ORDER BY created_at DESC LIMIT 500').fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]


def _float_or_none(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except Exception:
        return None


@app.get('/api/admin/audio/{upload_id}')
def admin_audio(upload_id: str, token: str = Query(...)):
    """
    Admin panelinde sesi dinletirken artık tüm dosyayı değil,
    kullanıcının seçtiği aralığı dinletir.

    Örnek:
    Kullanıcı 5.5 - 11.2 sn seçtiyse admin audio player sadece o bölümü çalar.
    Seçili aralık yoksa eski davranış korunur ve tüm dosya döner.
    """
    check_admin(token)
    conn = connect()
    row = conn.execute('SELECT * FROM uploads WHERE id=?', (upload_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, 'Kayıt bulunamadı.')

    p = Path(row['stored_path'])
    if not p.exists():
        p = Path('/app') / str(row['stored_path']).lstrip('/')

    if not p.exists():
        raise HTTPException(404, 'Ses dosyası bulunamadı.')

    start_sec = _float_or_none(row['selected_start_sec'] if 'selected_start_sec' in row.keys() else None)
    end_sec = _float_or_none(row['selected_end_sec'] if 'selected_end_sec' in row.keys() else None)

    # Seçili aralık varsa admin tarafına sadece o kırpılmış WAV dosyasını gönder.
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        preview_dir = UPLOAD_DIR / '_admin_selected_ranges'
        preview_dir.mkdir(parents=True, exist_ok=True)

        safe_range = f'{start_sec:.2f}_{end_sec:.2f}'.replace('.', 'p')
        preview_path = preview_dir / f'{upload_id}_{safe_range}.wav'

        # Daha önce üretilmediyse veya orijinal dosya daha yeniyse yeniden üret.
        if (not preview_path.exists()) or (preview_path.stat().st_mtime < p.stat().st_mtime):
            save_audio_range(p, preview_path, start_sec=start_sec, end_sec=end_sec)

        return FileResponse(
            preview_path,
            media_type='audio/wav',
            filename=f'{Path(row["filename"]).stem}_selected_range.wav',
        )

    return FileResponse(p)


@app.post('/api/admin/records/{upload_id}/label')
def save_admin_label(upload_id: str, inp: AdminLabelIn, token: str = Query(...)):
    check_admin(token)

    if inp.admin_label not in CLASSES:
        raise HTTPException(400, 'Geçersiz sınıf.')

    conn = connect()
    cur = conn.execute(
        'UPDATE uploads SET admin_label=?, status=? WHERE id=?',
        (inp.admin_label, 'admin_labeled', upload_id),
    )
    conn.commit()
    conn.close()

    if cur.rowcount == 0:
        raise HTTPException(404, 'Kayıt bulunamadı.')

    return {'ok': True, 'message': 'Admin etiketi kaydedildi.'}


@app.post('/api/admin/retrain')
def start_retrain(inp: RetrainIn, token: str = Query(...)):
    check_admin(token)
    st = read_state()

    if st.get('training_state') == 'running' or st.get('yamnet_state') == 'running':
        raise HTTPException(409, 'Zaten bir eğitim işlemi çalışıyor.')

    job_id = str(uuid.uuid4())
    params = {'mode': inp.mode, 'record_ids': inp.record_ids}

    conn = connect()
    conn.execute(
        'INSERT INTO jobs(id,type,state,message,params_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?)',
        (
            job_id,
            'cnn_retrain',
            'pending',
            'CNN retraining sıraya alındı.',
            json.dumps(params, ensure_ascii=False),
            now(),
            now(),
        ),
    )
    conn.commit()
    conn.close()

    update_state(
        training_state='running',
        training_message='CNN retraining başlatıldı. Eski aktif model tahmin vermeye devam eder.',
    )
    return {'ok': True, 'job_id': job_id, 'message': 'CNN retraining başlatıldı.'}


@app.post('/api/admin/yamnet/train')
def start_yamnet(inp: RetrainIn, token: str = Query(...)):
    check_admin(token)
    st = read_state()

    if st.get('training_state') == 'running' or st.get('yamnet_state') == 'running':
        raise HTTPException(409, 'Zaten bir eğitim işlemi çalışıyor.')

    job_id = str(uuid.uuid4())
    params = {'mode': inp.mode, 'record_ids': inp.record_ids}

    conn = connect()
    conn.execute(
        'INSERT INTO jobs(id,type,state,message,params_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?)',
        (
            job_id,
            'yamnet_train',
            'pending',
            'YAMNet transfer eğitimi sıraya alındı.',
            json.dumps(params, ensure_ascii=False),
            now(),
            now(),
        ),
    )
    conn.commit()
    conn.close()

    update_state(
        yamnet_state='running',
        yamnet_message='YAMNet transfer eğitimi başlatıldı. Eski aktif model tahmin vermeye devam eder.',
    )
    return {'ok': True, 'job_id': job_id, 'message': 'YAMNet transfer eğitimi başlatıldı.'}
