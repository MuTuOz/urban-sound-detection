from __future__ import annotations
import json
import time
import traceback
from .db import init_db, connect, now
from .ml.modeling import ensure_initial_model
from .ml.training import train_cnn_job, train_yamnet_like_job


def claim_next_job():
    conn = connect()
    row = conn.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY created_at LIMIT 1").fetchone()
    if not row:
        conn.close(); return None
    conn.execute("UPDATE jobs SET state=?, message=?, updated_at=? WHERE id=?", ('running', 'İş başladı.', now(), row['id']))
    conn.commit()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (row['id'],)).fetchone()
    conn.close()
    return dict(row)

def finish_job(job_id, state, message, result):
    conn = connect()
    conn.execute("UPDATE jobs SET state=?, message=?, result_json=?, updated_at=? WHERE id=?", (state, message, json.dumps(result, ensure_ascii=False), now(), job_id))
    conn.commit(); conn.close()

def main():
    init_db()
    ensure_initial_model()
    print('Akıncı worker hazır.', flush=True)
    while True:
        job = claim_next_job()
        if not job:
            time.sleep(2)
            continue
        params = json.loads(job.get('params_json') or '{}')
        try:
            if job['type'] == 'cnn_retrain':
                result = train_cnn_job(params)
            elif job['type'] == 'yamnet_train':
                result = train_yamnet_like_job(params)
            else:
                result = {'ok': False, 'message': f'Bilinmeyen iş tipi: {job["type"]}'}
            finish_job(job['id'], 'finished' if result.get('ok') else 'failed', result.get('message', 'Tamamlandı.'), result)
        except Exception as e:
            traceback.print_exc()
            finish_job(job['id'], 'failed', str(e), {'ok': False, 'error': str(e)})
        time.sleep(1)

if __name__ == '__main__':
    main()
