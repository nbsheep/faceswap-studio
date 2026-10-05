/* FaceSwap Studio 前端逻辑
 * 只依赖原生 API，无构建步骤。
 */

const API = '';

const state = {
  sourceFile: null,
  targetFile: null,
  targetFaces: 0,
  targetFaceIndex: 0,
  result: null,        // { data_uri, url, meta, elapsed_ms }
  engines: [],
  currentEngine: '',
};

const $ = (id) => document.getElementById(id);

const els = {
  dropSource: $('dropSource'), dropTarget: $('dropTarget'),
  fileSource: $('fileSource'), fileTarget: $('fileTarget'),
  prevSource: $('prevSource'), prevTarget: $('prevTarget'),
  facePicker: $('facePicker'), faceChips: $('faceChips'),
  blend: $('optBlend'), feather: $('optFeather'), scale: $('optScale'),
  strength: $('optStrength'), color: $('optColor'), enhance: $('optEnhance'),
  valFeather: $('valFeather'), valScale: $('valScale'), valStrength: $('valStrength'),
  btnSwap: $('btnSwap'), btnDetect: $('btnDetect'), btnDownload: $('btnDownload'),
  btnReset: $('btnReset'), status: $('status'),
  stack: $('stack'), imgTarget: $('imgTarget'), resultImg: $('resultImg'),
  resultEmpty: $('resultEmpty'), compareRow: $('compareRow'),
  compareRange: $('compareRange'), splitLine: $('splitLine'), meta: $('meta'),
  engineBadge: $('engineBadge'), engineName: $('engineName'),
  historySection: $('historySection'), gallery: $('gallery'),
};

/* ------------------------------------------------------------ 初始化 */

async function init() {
  bindUpload(els.dropSource, els.fileSource, 'source');
  bindUpload(els.dropTarget, els.fileTarget, 'target');
  bindSlider(els.feather, els.valFeather);
  bindSlider(els.scale, els.valScale, (v) => (v / 100).toFixed(2));
  bindSlider(els.strength, els.valStrength);

  document.querySelectorAll('[data-clear]').forEach((b) => {
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      clearSlot(b.dataset.clear);
    });
  });

  els.btnSwap.addEventListener('click', doSwap);
  els.btnDetect.addEventListener('click', doDetect);
  els.btnDownload.addEventListener('click', download);
  els.btnReset.addEventListener('click', resetAll);
  els.compareRange.addEventListener('input', applyCompare);

  await loadEngines();
  await loadHistory();
}

async function loadEngines() {
  try {
    const res = await fetch(API + '/api/engines');
    state.engines = await res.json();
    const usable = state.engines.filter((e) => e.available);
    // 和后端 auto_engine() 保持同一优先级：深度 > insightface > 经典
    const priority = ['deep', 'insightface', 'classic'];
    const best = priority.map((n) => usable.find((e) => e.name === n)).find(Boolean);
    if (!best) throw new Error('没有可用引擎');

    state.currentEngine = best.name;
    els.engineName.textContent = best.display_name;
    // 只有退化成「经典几何法」才算降级模式，深度引擎是正常状态
    els.engineBadge.classList.toggle('down', best.name === 'classic');
    els.engineBadge.title = state.engines
      .map((e) => `${e.display_name}: ${e.available ? '可用' : e.reason}`)
      .join('\n');
  } catch (err) {
    els.engineName.textContent = '引擎加载失败';
    els.engineBadge.classList.add('down');
    setStatus('无法连接后端服务，请确认 python backend/app.py 正在运行。', 'error');
  }
}

/* ------------------------------------------------------------ 上传 */

function bindUpload(drop, input, slot) {
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files[0]) setSlot(slot, input.files[0]);
  });

  ['dragenter', 'dragover'].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add('over');
    })
  );
  ['dragleave', 'drop'].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove('over');
    })
  );
  drop.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) setSlot(slot, f);
  });
}

async function setSlot(slot, file) {
  const drop = slot === 'source' ? els.dropSource : els.dropTarget;
  const img = slot === 'source' ? els.prevSource : els.prevTarget;

  img.src = URL.createObjectURL(file);
  drop.classList.add('filled');
  state[slot === 'source' ? 'sourceFile' : 'targetFile'] = file;

  if (slot === 'target') {
    state.targetFaceIndex = 0;
    await refreshTargetFaces(file);
  }
  syncButtons();
}

function clearSlot(slot) {
  const drop = slot === 'source' ? els.dropSource : els.dropTarget;
  const img = slot === 'source' ? els.prevSource : els.prevTarget;
  img.removeAttribute('src');
  drop.classList.remove('filled');

  if (slot === 'source') {
    state.sourceFile = null;
    els.fileSource.value = '';
  } else {
    state.targetFile = null;
    state.targetFaces = 0;
    els.fileTarget.value = '';
    els.facePicker.hidden = true;
    els.faceChips.innerHTML = '';
  }
  syncButtons();
  setStatus('上传两张图后即可开始。');
}

async function refreshTargetFaces(file) {
  const form = new FormData();
  form.append('image', file);
  try {
    const res = await fetch(API + '/api/detect', { method: 'POST', body: form });
    if (!res.ok) throw new Error('检测失败');
    const data = await res.json();
    state.targetFaces = data.faces.length;

    if (data.faces.length > 1) {
      els.faceChips.innerHTML = '';
      data.faces.forEach((f, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip' + (i === 0 ? ' active' : '');
        b.textContent = i + 1;
        b.title = `位置 (${f.x}, ${f.y})  尺寸 ${f.w}×${f.h}`;
        b.addEventListener('click', () => {
          state.targetFaceIndex = i;
          [...els.faceChips.children].forEach((c) => c.classList.remove('active'));
          b.classList.add('active');
        });
        els.faceChips.appendChild(b);
      });
      els.facePicker.hidden = false;
    } else {
      els.facePicker.hidden = true;
    }

    if (data.faces.length === 0) {
      setStatus(`目标图里没检测到人脸（${data.elapsed_ms}ms）。换一张更清晰的正脸照。`, 'error');
    } else {
      setStatus(`检测到 ${data.faces.length} 张脸，耗时 ${data.elapsed_ms}ms。可以开始换脸了。`, 'ok');
    }
  } catch (err) {
    state.targetFaces = 0;
    setStatus('人脸检测失败：' + err.message, 'error');
  }
  syncButtons();
}

/* ------------------------------------------------------------ 换脸 */

function collectOptions() {
  return {
    target_face_index: state.targetFaceIndex,
    source_face_index: 0,
    feather: Number(els.feather.value),
    mask_scale: Number(els.scale.value) / 100,
    color_match: els.color.checked,
    blend_mode: els.blend.value,
    strength: Number(els.strength.value),
    enhance: els.enhance.checked,
  };
}

async function doSwap() {
  if (!state.sourceFile || !state.targetFile) return;

  const form = new FormData();
  form.append('source', state.sourceFile);
  form.append('target', state.targetFile);
  form.append('options', JSON.stringify(collectOptions()));
  form.append('engine', state.currentEngine);

  setBusy(true, '正在换脸…');
  const t0 = performance.now();
  try {
    const res = await fetch(API + '/api/swap', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '换脸失败');

    state.result = data;
    showResult(data);
    const ms = Math.round(performance.now() - t0);
    setStatus(`完成！用时 ${ms}ms（后端 ${data.elapsed_ms}ms）。拖动下方滑块可对比原图。`, 'ok');
    loadHistory();
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    setBusy(false);
  }
}

async function doDetect() {
  if (!state.targetFile) return;
  const form = new FormData();
  form.append('image', state.targetFile);
  setBusy(true, '正在检测人脸…');
  try {
    const res = await fetch(API + '/api/detect', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '检测失败');

    const box = data.faces[state.targetFaceIndex] || data.faces[0];
    if (!box) throw new Error('没有检测到人脸');

    // 把检测结果画在原图上，直观看到「眼睛/嘴」对齐点
    const img = new Image();
    img.src = els.prevTarget.src;
    await img.decode();
    const canvas = document.createElement('canvas');
    canvas.width = data.width;
    canvas.height = data.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, data.width, data.height);

    ctx.lineWidth = Math.max(2, data.width / 320);
    ctx.strokeStyle = '#3b6ef5';
    ctx.strokeRect(box.x, box.y, box.w, box.h);

    const colors = { left_eye: '#12a150', right_eye: '#12a150', mouth: '#dc2626', nose: '#d97706' };
    Object.entries(box.landmarks || {}).forEach(([name, pt]) => {
      if (name.startsWith('_')) return;
      const cx = box.x + pt[0] * box.w;
      const cy = box.y + pt[1] * box.h;
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(4, data.width / 130), 0, Math.PI * 2);
      ctx.fillStyle = colors[name] || '#7b5cf0';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    data.faces.forEach((f, i) => {
      ctx.fillStyle = i === state.targetFaceIndex ? 'rgba(59,110,245,.16)' : 'rgba(107,114,128,.10)';
      ctx.fillRect(f.x, f.y, f.w, f.h);
    });

    state.result = null;
    els.resultEmpty.hidden = true;
    els.stack.hidden = false;
    els.resultImg.src = canvas.toDataURL('image/jpeg', 0.92);
    els.imgTarget.src = els.prevTarget.src;
    els.compareRange.value = 100;
    els.resultImg.style.clipPath = '';
    els.splitLine.hidden = true;
    els.compareRow.hidden = true;
    els.meta.hidden = true;
    els.btnDownload.disabled = true;
    els.btnReset.disabled = false;
    setStatus(`检测到 ${data.faces.length} 张脸。蓝框=人脸，绿点=眼睛，红点=嘴（这些点决定对齐精度）。`, 'ok');
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    setBusy(false);
  }
}

function showResult(data) {
  els.resultEmpty.hidden = true;
  els.stack.hidden = false;
  els.imgTarget.src = data.target_data_uri;
  els.resultImg.src = data.result_data_uri;

  // 重新计算裁切（图片尺寸可能变化）
  requestAnimationFrame(() => {
    els.compareRange.value = 100;
    applyCompare();
  });
  els.compareRow.hidden = false;
  els.splitLine.hidden = false;
  els.btnDownload.disabled = false;
  els.btnReset.disabled = false;

  const m = data.meta || {};
  const chips = [
    `引擎 <b>${data.engine}</b>`,
    `耗时 <b>${data.elapsed_ms}ms</b>`,
    `目标检出 <b>${m.target_faces ?? '-'}</b> 张脸`,
    `模型 <b>${m.model ? 'inswapper_128' : '内置几何法'}</b>`,
  ];
  els.meta.innerHTML = chips.map((c) => `<span>${c}</span>`).join('');
  els.meta.hidden = false;
}

/* 对比滑块：上层（结果）裁切到左侧 v%，右侧露出原图 */
function applyCompare() {
  const v = Number(els.compareRange.value);
  els.resultImg.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  els.splitLine.style.left = v + '%';
  els.splitLine.hidden = v >= 100 || v <= 0;
}

/* ------------------------------------------------------------ 历史 */

async function loadHistory() {
  try {
    const res = await fetch(API + '/api/history?limit=12');
    const items = await res.json();
    if (!items.length) {
      els.historySection.hidden = true;
      return;
    }
    els.historySection.hidden = false;
    els.gallery.innerHTML = '';
    items.forEach((it) => {
      const fig = document.createElement('figure');
      const time = new Date(it.created_at).toLocaleTimeString('zh-CN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      });
      fig.innerHTML = `
        <img src="${it.result_url}" alt="结果" loading="lazy">
        <figcaption><span>${it.engine}</span><span>${it.elapsed_ms}ms · ${time}</span></figcaption>`;
      fig.querySelector('img').addEventListener('click', () => {
        els.resultEmpty.hidden = true;
        els.stack.hidden = false;
        els.imgTarget.src = it.target_url;
        els.resultImg.src = it.result_url;
        els.compareRange.value = 100;
        applyCompare();
        els.compareRow.hidden = false;
        els.splitLine.hidden = true;
        els.meta.innerHTML = `<span>历史结果</span><span>引擎 <b>${it.engine}</b></span><span>耗时 <b>${it.elapsed_ms}ms</b></span>`;
        els.meta.hidden = false;
        els.btnDownload.disabled = false;
        els.btnReset.disabled = false;
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
      els.gallery.appendChild(fig);
    });
  } catch (_) {
    /* 历史加载失败不影响主流程 */
  }
}

/* ------------------------------------------------------------ 杂项 */

function download() {
  if (!state.result) return;
  const a = document.createElement('a');
  a.href = state.result.result_url;
  a.download = 'faceswap_' + state.result.id;
  a.click();
}

function resetAll() {
  clearSlot('source');
  clearSlot('target');
  els.stack.hidden = true;
  els.resultEmpty.hidden = false;
  els.compareRow.hidden = true;
  els.splitLine.hidden = true;
  els.meta.hidden = true;
  els.btnDownload.disabled = true;
  els.btnReset.disabled = true;
  state.result = null;
  setStatus('上传两张图后即可开始。');
}

function syncButtons() {
  const ok = Boolean(state.sourceFile && state.targetFile);
  els.btnSwap.disabled = !ok || !state.currentEngine;
  els.btnDetect.disabled = !state.targetFile;
}

function bindSlider(input, label, fmt = (v) => v) {
  const sync = () => (label.textContent = fmt(Number(input.value)));
  input.addEventListener('input', sync);
  sync();
}

function setBusy(busy, text) {
  els.btnSwap.disabled = busy || !(state.sourceFile && state.targetFile);
  els.btnDetect.disabled = busy || !state.targetFile;
  if (busy) {
    els.btnSwap.innerHTML = `<span class="spinner"></span>${text || '处理中…'}`;
    setStatus(text || '处理中…', 'busy');
  } else {
    els.btnSwap.textContent = '开始换脸';
  }
}

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.className = 'tip' + (kind ? ' ' + kind : '');
}

init();
