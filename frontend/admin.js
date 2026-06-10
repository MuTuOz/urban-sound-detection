const $ = (id) => document.getElementById(id);

const tokenInput = $('tokenInput');
const saveToken = $('saveToken');
const tokenMsg = $('tokenMsg');
const activeModel = $('activeModel');
const cnnState = $('cnnState');
const lastModel = $('lastModel');
const yamnetState = $('yamnetState');
const statusMsg = $('statusMsg');
const busyLabel = $('busyLabel');
const recordsEl = $('records');
const refreshBtn = $('refresh');
const selectAllBtn = $('selectAll');
const clearSelBtn = $('clearSel');
const retrainSelectedBtn = $('retrainSelected');
const retrainAllBtn = $('retrainAll');
const yamnetTrainBtn = $('yamnetTrain');

let classesCache = [];
let recordsCache = [];

tokenInput.value = localStorage.getItem('akincises_admin_token') || '';

function token() { return tokenInput.value.trim(); }
function authUrl(path) { return `${path}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(token())}`; }
function fmt(n) { const x = Number(n); return Number.isFinite(x) ? x.toFixed(1) : '-'; }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

async function loadClasses() {
  try {
    const r = await fetch('/api/classes');
    classesCache = await r.json();
  } catch (_) { classesCache = []; }
}

function labelTr(label) {
  const c = classesCache.find(x => x.label === label);
  return c ? (c.label_tr || c.label) : (label || '-');
}

function classOptions(selected = '') {
  return classesCache.map(c => `<option value="${esc(c.label)}" ${c.label === selected ? 'selected' : ''}>${esc(c.label_tr || c.label)}</option>`).join('');
}

function selectedIds() {
  return Array.from(document.querySelectorAll('.record-check:checked')).map(x => x.value);
}

async function pollStatus() {
  try {
    const r = await fetch('/api/model/status?ts=' + Date.now(), { cache: 'no-store' });
    const s = await r.json();
    activeModel.textContent = s.active_version || '-';
    cnnState.textContent = s.training_state || '-';
    lastModel.textContent = s.last_finished_version || '-';
    yamnetState.textContent = s.yamnet_state || '-';

    const cnnRunning = s.training_state === 'running';
    const yamRunning = s.yamnet_state === 'running';
    busyLabel.classList.toggle('show', cnnRunning || yamRunning);
    busyLabel.querySelector('span:last-child').textContent = cnnRunning && yamRunning
      ? 'CNN retraining ve YAMNet eğitimi çalışıyor...'
      : cnnRunning
        ? 'CNN retraining çalışıyor...'
        : yamRunning
          ? 'YAMNet transfer eğitimi çalışıyor...'
          : 'Model eğitimi çalışıyor...';

    statusMsg.textContent = [s.training_message, s.yamnet_message].filter(Boolean).join(' | ');
    retrainSelectedBtn.classList.toggle('as-working', cnnRunning);
    retrainAllBtn.classList.toggle('as-working', cnnRunning);
    yamnetTrainBtn.classList.toggle('as-working', yamRunning);
  } catch (_) {}
}

async function loadRecords() {
  if (!token()) {
    recordsEl.innerHTML = '<p class="notice">Önce admin token girin.</p>';
    return;
  }
  recordsEl.innerHTML = '<p class="notice"><span class="spinner"></span> Kayıtlar yükleniyor...</p>';
  try {
    const r = await fetch(authUrl('/api/admin/records'), { cache: 'no-store' });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Kayıtlar alınamadı.');
    recordsCache = data;
    renderRecords();
  } catch (e) {
    recordsEl.innerHTML = `<p class="error">${esc(e.message || 'Kayıtlar alınamadı.')}</p>`;
  }
}

function probabilitiesHtml(record) {
  const probs = Array.isArray(record.probabilities) ? record.probabilities : [];
  if (!probs.length) return '';
  return `
    <details class="prob-list">
      <summary>Tahmin yüzdelerini göster</summary>
      ${probs.slice(0, 8).map(p => `<div class="prob-mini"><span>${esc(p.label_tr || p.label)}</span><b>${esc(p.percent)}%</b></div>`).join('')}
    </details>
  `;
}

function rangeText(record) {
  const s = record.selected_start_sec;
  const e = record.selected_end_sec;
  if (s === null || s === undefined || e === null || e === undefined) return 'Aralık: tüm ses / eski kayıt';
  return `Aralık: ${fmt(s)} - ${fmt(e)} sn`;
}

function renderRecords() {
  if (!recordsCache.length) {
    recordsEl.innerHTML = '<p class="notice">Henüz kayıt yok.</p>';
    return;
  }
  recordsEl.innerHTML = recordsCache.map(r => {
    const adminValue = r.admin_label || r.user_label || r.model_prediction || '';
    return `
      <article class="record" data-id="${esc(r.id)}">
        <div class="record-head">
          <label class="row"><input class="checkbox record-check" type="checkbox" value="${esc(r.id)}"><span class="record-title">${esc(r.filename)}</span></label>
          <span class="pill">${esc(r.status || 'pending')}</span>
        </div>
        <div class="record-meta">
          <span class="range-pill">${esc(rangeText(r))}</span>
          <span class="range-pill">Tahmin: ${esc(labelTr(r.model_prediction))}</span>
          <span class="range-pill">Kullanıcı: ${esc(labelTr(r.user_label))}</span>
          <span class="range-pill">Admin: ${esc(labelTr(r.admin_label))}</span>
        </div>
        <audio controls src="${authUrl('/api/admin/audio/' + encodeURIComponent(r.id))}"></audio>
        ${probabilitiesHtml(r)}
        <div class="record-actions">
          <select class="admin-label">${classOptions(adminValue)}</select>
          <button class="btn save-label" type="button">Admin etiketini kaydet</button>
          <span class="notice save-msg"></span>
        </div>
      </article>
    `;
  }).join('');

  recordsEl.querySelectorAll('.save-label').forEach(btn => {
    btn.addEventListener('click', async () => {
      const rec = btn.closest('.record');
      const id = rec.dataset.id;
      const label = rec.querySelector('.admin-label').value;
      const msg = rec.querySelector('.save-msg');
      msg.textContent = 'Kaydediliyor...';
      try {
        const res = await fetch(authUrl(`/api/admin/records/${encodeURIComponent(id)}/label`), {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({admin_label: label})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Kaydedilemedi.');
        msg.textContent = data.message || 'Kaydedildi.';
        await loadRecords();
      } catch (e) {
        msg.textContent = e.message || 'Kaydedilemedi.';
      }
    });
  });
}

async function startJob(url, payload) {
  if (!token()) {
    tokenMsg.textContent = 'Önce admin token girin.';
    return;
  }
  try {
    const r = await fetch(authUrl(url), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || data.message || 'İşlem başlatılamadı.');
    statusMsg.textContent = data.message || 'İşlem başlatıldı.';
    await pollStatus();
  } catch (e) {
    statusMsg.textContent = e.message || 'İşlem başlatılamadı.';
  }
}

saveToken.addEventListener('click', async () => {
  localStorage.setItem('akincises_admin_token', token());
  tokenMsg.textContent = 'Token kaydedildi.';
  await loadRecords();
  await pollStatus();
});
refreshBtn.addEventListener('click', loadRecords);
selectAllBtn.addEventListener('click', () => document.querySelectorAll('.record-check').forEach(x => x.checked = true));
clearSelBtn.addEventListener('click', () => document.querySelectorAll('.record-check').forEach(x => x.checked = false));
retrainSelectedBtn.addEventListener('click', () => startJob('/api/admin/retrain', {mode: 'selected', record_ids: selectedIds()}));
retrainAllBtn.addEventListener('click', () => startJob('/api/admin/retrain', {mode: 'all', record_ids: []}));
yamnetTrainBtn.addEventListener('click', () => startJob('/api/admin/yamnet/train', {mode: 'all', record_ids: []}));

(async function init() {
  await loadClasses();
  await pollStatus();
  await loadRecords();
  setInterval(pollStatus, 2500);
})();
