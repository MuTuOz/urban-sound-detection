const $ = (id) => document.getElementById(id);

const fileInput = $('fileInput');
const dropZone = $('dropZone');
const fileName = $('fileName');
const predictBtn = $('predictBtn');
const resetBtn = $('resetBtn');
const recordBtn = $('recordBtn');
const processing = $('processing');
const errorBox = $('error');
const resultBlock = $('resultBlock');
const rangeBlock = $('rangeBlock');
const previewAudio = $('previewAudio');
const durationBadge = $('durationBadge');
const startSec = $('startSec');
const endSec = $('endSec');
const startLabel = $('startLabel');
const endLabel = $('endLabel');
const selectedDurationLabel = $('selectedDurationLabel');
const useFirst5 = $('useFirst5');
const useCurrent5 = $('useCurrent5');
const playSelected = $('playSelected');
const rangeHint = $('rangeHint');
const rangeTrack = $('rangeTrack');
const rangeSelection = $('rangeSelection');
const leftHandle = $('leftHandle');
const rightHandle = $('rightHandle');
const rangeMidLabel = $('rangeMidLabel');
const rangeEndLabel = $('rangeEndLabel');

let selectedFile = null;
let objectUrl = null;
let audioDuration = 0;
let rangeStart = 0;
let rangeEnd = 5;
let draggingHandle = null;
let mediaRecorder = null;
let chunks = [];
let classesCache = [];
let selectedPreviewActive = false;

const DEFAULT_RANGE_SECONDS = 5;
const MIN_RANGE_SECONDS = 2;

function fmt(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return '0.0';
  return x.toFixed(1);
}

function fmtClock(n) {
  const x = Math.max(0, Number(n) || 0);
  const m = Math.floor(x / 60);
  const s = Math.floor(x % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function getMaxTime() {
  if (audioDuration > 0) return audioDuration;
  return Math.max(DEFAULT_RANGE_SECONDS, rangeEnd || DEFAULT_RANGE_SECONDS);
}

function stopSelectedPreview(resetToStart = false) {
  selectedPreviewActive = false;
  playSelected.textContent = 'Seçili aralığı dinle';
  playSelected.classList.remove('as-working');
  if (resetToStart && Number.isFinite(rangeStart)) {
    previewAudio.currentTime = rangeStart;
  }
}

function normalizeRange(start, end) {
  const maxTime = getMaxTime();
  let s = Number(start);
  let e = Number(end);

  if (!Number.isFinite(s)) s = 0;
  if (!Number.isFinite(e)) e = Math.min(DEFAULT_RANGE_SECONDS, maxTime);

  s = Math.max(0, Math.min(s, Math.max(0, maxTime)));
  e = Math.max(0, Math.min(e, Math.max(0, maxTime)));

  const minLen = Math.min(MIN_RANGE_SECONDS, Math.max(0.1, maxTime));

  if (e - s < minLen) {
    if (draggingHandle === 'left') {
      s = e - minLen;
    } else {
      e = s + minLen;
    }
  }

  if (s < 0) {
    e = Math.min(maxTime, e - s);
    s = 0;
  }

  if (e > maxTime) {
    s = Math.max(0, s - (e - maxTime));
    e = maxTime;
  }

  if (e <= s) {
    e = Math.min(maxTime, s + minLen);
  }

  return { start: s, end: e };
}

function updateRangeUI() {
  const maxTime = Math.max(0.1, getMaxTime());
  const startPct = Math.max(0, Math.min(100, (rangeStart / maxTime) * 100));
  const endPct = Math.max(0, Math.min(100, (rangeEnd / maxTime) * 100));
  const widthPct = Math.max(0, endPct - startPct);

  rangeSelection.style.left = `${startPct}%`;
  rangeSelection.style.width = `${widthPct}%`;
  leftHandle.style.left = `${startPct}%`;
  rightHandle.style.left = `${endPct}%`;

  startSec.value = String(rangeStart);
  endSec.value = String(rangeEnd);

  startLabel.textContent = `${fmt(rangeStart)} sn`;
  endLabel.textContent = `${fmt(rangeEnd)} sn`;
  selectedDurationLabel.textContent = `${fmt(rangeEnd - rangeStart)} sn`;
  rangeHint.textContent = `Seçilen aralık: ${fmt(rangeStart)} - ${fmt(rangeEnd)} sn`;

  leftHandle.setAttribute('aria-valuemin', '0');
  leftHandle.setAttribute('aria-valuemax', fmt(rangeEnd));
  leftHandle.setAttribute('aria-valuenow', fmt(rangeStart));
  rightHandle.setAttribute('aria-valuemin', fmt(rangeStart));
  rightHandle.setAttribute('aria-valuemax', fmt(maxTime));
  rightHandle.setAttribute('aria-valuenow', fmt(rangeEnd));

  rangeMidLabel.textContent = fmtClock(maxTime / 2);
  rangeEndLabel.textContent = fmtClock(maxTime);
}

function setRange(start, end) {
  const r = normalizeRange(start, end);
  rangeStart = r.start;
  rangeEnd = r.end;
  updateRangeUI();

  if (selectedPreviewActive) {
    previewAudio.currentTime = rangeStart;
  }

  return { start: rangeStart, end: rangeEnd };
}

function setDefaultRange() {
  const maxTime = getMaxTime();
  setRange(0, Math.min(DEFAULT_RANGE_SECONDS, maxTime));
}

function valueFromPointerEvent(e) {
  const rect = rangeTrack.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
  return pct * getMaxTime();
}

function beginDrag(handleName, e) {
  if (!selectedFile) return;
  draggingHandle = handleName;
  rangeTrack.classList.add('dragging-range');
  try {
    rangeTrack.setPointerCapture(e.pointerId);
  } catch (_) {}
  e.preventDefault();
}

function dragTo(e) {
  if (!draggingHandle) return;
  const value = valueFromPointerEvent(e);

  if (draggingHandle === 'left') {
    setRange(value, rangeEnd);
  } else {
    setRange(rangeStart, value);
  }
}

function endDrag(e) {
  if (!draggingHandle) return;
  draggingHandle = null;
  rangeTrack.classList.remove('dragging-range');
  try {
    rangeTrack.releasePointerCapture(e.pointerId);
  } catch (_) {}
}

function nudgeHandle(handleName, amount) {
  draggingHandle = handleName;
  if (handleName === 'left') {
    setRange(rangeStart + amount, rangeEnd);
  } else {
    setRange(rangeStart, rangeEnd + amount);
  }
  draggingHandle = null;
}

function clearSelectedFile() {
  selectedFile = null;
  audioDuration = 0;
  rangeStart = 0;
  rangeEnd = DEFAULT_RANGE_SECONDS;
  draggingHandle = null;
  selectedPreviewActive = false;

  fileInput.value = '';
  fileName.textContent = '';
  predictBtn.disabled = true;
  rangeBlock.classList.add('hidden');
  resultBlock.classList.remove('show');
  resultBlock.innerHTML = '';
  errorBox.textContent = '';
  processing.classList.remove('show');
  playSelected.textContent = 'Seçili aralığı dinle';
  playSelected.classList.remove('as-working');

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  previewAudio.removeAttribute('src');
  previewAudio.load();
  updateRangeUI();
}

function setSelectedFile(file) {
  if (!file) return;

  selectedFile = file;
  audioDuration = 0;
  fileName.textContent = file.name;
  predictBtn.disabled = false;
  errorBox.textContent = '';
  resultBlock.classList.remove('show');
  resultBlock.innerHTML = '';
  stopSelectedPreview(false);

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  previewAudio.src = objectUrl;
  previewAudio.load();
  rangeBlock.classList.remove('hidden');
  durationBadge.textContent = 'Süre: okunuyor...';
  setRange(0, DEFAULT_RANGE_SECONDS);
}

async function loadClasses() {
  try {
    const r = await fetch('/api/classes');
    classesCache = await r.json();
  } catch (_) {
    classesCache = [];
  }
}

function classOptions(selected = '') {
  return classesCache.map(c => `<option value="${c.label}" ${c.label === selected ? 'selected' : ''}>${c.label_tr || c.label}</option>`).join('');
}

function renderResult(data) {
  const prediction = data.prediction || {};
  const range = data.selected_range || {};
  const rows = (data.probabilities || []).map(p => `
    <div class="bar-row">
      <div class="bar-top"><span>${p.label_tr || p.label}</span><span>${p.percent}%</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, p.percent || 0))}%"></div></div>
    </div>
  `).join('');

  resultBlock.innerHTML = `
    <div class="result-hero">
      <small>${data.decision_mode || 'CNN'} · ${data.model_version || '-'}</small>
      <h2>${prediction.label_tr || prediction.label || '-'}</h2>
      <p>En güçlü tahmin: <b>${prediction.percent ?? '-'}%</b></p>
      <p>Analiz edilen aralık: <b>${fmt(range.start_sec || rangeStart)} - ${fmt(range.end_sec || rangeEnd)} sn</b></p>
    </div>
    ${rows}
    <div class="feedback">
      <div class="kicker">3. Aşama</div>
      <h2>Aslında bu ses neydi?</h2>
      <p class="muted">Doğru etiketi seçerseniz admin onayından sonra bu kayıt modele eğitim verisi olarak eklenir.</p>
      <select id="feedbackLabel">${classOptions(prediction.label)}</select>
      <button id="sendFeedback" class="btn" type="button">Geri bildirimi gönder</button>
      <div id="feedbackMsg" class="notice"></div>
    </div>
  `;
  resultBlock.classList.add('show');

  $('sendFeedback').addEventListener('click', async () => {
    const label = $('feedbackLabel').value;
    const msg = $('feedbackMsg');
    msg.textContent = 'Kaydediliyor...';
    try {
      const r = await fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({upload_id: data.upload_id, user_label: label})
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || j.message || 'Kayıt başarısız.');
      msg.textContent = j.message || 'Geri bildirim kaydedildi.';
    } catch (e) {
      msg.textContent = e.message || 'Geri bildirim kaydedilemedi.';
    }
  });
}

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('click', e => e.stopPropagation());
fileInput.addEventListener('change', () => setSelectedFile(fileInput.files && fileInput.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  setSelectedFile(file);
});

previewAudio.addEventListener('loadedmetadata', () => {
  audioDuration = Number.isFinite(previewAudio.duration) ? previewAudio.duration : 0;
  durationBadge.textContent = audioDuration > 0 ? `Süre: ${fmt(audioDuration)} sn` : 'Süre: bilinmiyor';
  setDefaultRange();
});

previewAudio.addEventListener('timeupdate', () => {
  if (selectedPreviewActive && previewAudio.currentTime >= rangeEnd) {
    previewAudio.pause();
    stopSelectedPreview(true);
  }
});

previewAudio.addEventListener('ended', () => stopSelectedPreview(false));

playSelected.addEventListener('click', async () => {
  if (!selectedFile) return;

  if (selectedPreviewActive) {
    previewAudio.pause();
    stopSelectedPreview(false);
    return;
  }

  try {
    previewAudio.currentTime = rangeStart;
    selectedPreviewActive = true;
    playSelected.textContent = 'Seçili aralığı durdur';
    playSelected.classList.add('as-working');
    await previewAudio.play();
  } catch (_) {
    stopSelectedPreview(false);
  }
});

leftHandle.addEventListener('pointerdown', (e) => beginDrag('left', e));
rightHandle.addEventListener('pointerdown', (e) => beginDrag('right', e));

rangeTrack.addEventListener('pointerdown', (e) => {
  if (e.target === leftHandle || e.target === rightHandle) return;
  if (!selectedFile) return;

  const value = valueFromPointerEvent(e);
  const nearest = Math.abs(value - rangeStart) <= Math.abs(value - rangeEnd) ? 'left' : 'right';
  beginDrag(nearest, e);
  dragTo(e);
});

rangeTrack.addEventListener('pointermove', dragTo);
rangeTrack.addEventListener('pointerup', endDrag);
rangeTrack.addEventListener('pointercancel', endDrag);

leftHandle.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 1 : 0.1;
  if (e.key === 'ArrowLeft') { e.preventDefault(); nudgeHandle('left', -step); }
  if (e.key === 'ArrowRight') { e.preventDefault(); nudgeHandle('left', step); }
});

rightHandle.addEventListener('keydown', (e) => {
  const step = e.shiftKey ? 1 : 0.1;
  if (e.key === 'ArrowLeft') { e.preventDefault(); nudgeHandle('right', -step); }
  if (e.key === 'ArrowRight') { e.preventDefault(); nudgeHandle('right', step); }
});

useFirst5.addEventListener('click', () => {
  stopSelectedPreview(false);
  setRange(0, Math.min(DEFAULT_RANGE_SECONDS, getMaxTime()));
});

useCurrent5.addEventListener('click', () => {
  stopSelectedPreview(false);
  const s = Number.isFinite(previewAudio.currentTime) ? previewAudio.currentTime : 0;
  setRange(s, Math.min(s + DEFAULT_RANGE_SECONDS, getMaxTime()));
});

resetBtn.addEventListener('click', clearSelectedFile);

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    errorBox.textContent = 'Önce bir ses dosyası seçin.';
    return;
  }

  stopSelectedPreview(false);
  const range = setRange(rangeStart, rangeEnd);
  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('start_sec', String(range.start));
  fd.append('end_sec', String(range.end));

  processing.classList.add('show');
  predictBtn.disabled = true;
  resetBtn.disabled = true;
  errorBox.textContent = '';

  try {
    const r = await fetch('/api/predict', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'Tahmin alınamadı.');
    renderResult(j);
    requestAnimationFrame(() => {
      resultBlock.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  } catch (e) {
    errorBox.textContent = e.message || 'Tahmin sırasında hata oluştu.';
  } finally {
    processing.classList.remove('show');
    predictBtn.disabled = false;
    resetBtn.disabled = false;
  }
});

recordBtn.addEventListener('click', async () => {
  try {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      recordBtn.textContent = 'Kaydı başlat';
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const file = new File([blob], `mikrofon_kaydi_${Date.now()}.webm`, { type: 'audio/webm' });
      setSelectedFile(file);
    };
    mediaRecorder.start();
    recordBtn.textContent = 'Kaydı durdur';
  } catch (e) {
    errorBox.textContent = 'Mikrofon erişimi alınamadı.';
  }
});

updateRangeUI();
loadClasses();
