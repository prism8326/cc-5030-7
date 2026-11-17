"""The panel's single-page UI, served at GET /.

Everything visible is driven by the interactive-beat kernel over SSE:
lane playheads, bar indices and hit flashes render kernel state, not local
animation; controls POST back into the kernel; the monitor stream is the
kernel's actual rendered audio; failures (packet loss, resync, rejections)
are reported back into the page's log from kernel events.
"""

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>interbeat 互动节拍面板</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c313a; --fg:#d7dce2;
          --dim:#8a919c; --a:#4fc3f7; --b:#aed581; --c:#ffb74d; --bad:#ef5350; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 "SF Mono", ui-monospace, Menlo, Consolas, monospace; }
  header { display:flex; gap:18px; align-items:center; flex-wrap:wrap;
           padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
  header .big { font-size:20px; min-width:130px; }
  header .grp { display:flex; gap:8px; align-items:center; }
  button, select { background:#262b33; color:var(--fg); border:1px solid var(--line);
         border-radius:4px; padding:5px 12px; font:inherit; cursor:pointer; }
  button:hover { border-color:var(--a); }
  button.warn { border-color:var(--bad); color:var(--bad); }
  input[type=range] { width:120px; }
  main { display:grid; grid-template-columns: 1fr 380px; gap:12px; padding:12px 16px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; }
  canvas { width:100%; display:block; }
  table { width:100%; border-collapse:collapse; }
  td, th { padding:2px 6px; text-align:right; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:normal; text-align:right; }
  #log { height:180px; overflow-y:auto; font-size:12px; white-space:pre-wrap; }
  #log .bad { color:var(--bad); }
  .muted { color:var(--dim); }
  .pill { padding:1px 8px; border-radius:10px; border:1px solid var(--line); }
  .pill.on { border-color:var(--b); color:var(--b); }
  h3 { margin:4px 0 8px; font-size:13px; color:var(--dim); font-weight:normal; }
  .steps { display:flex; gap:2px; margin:6px 0 2px; }
  .step { width:16px; height:22px; background:#262b33; border:1px solid var(--line);
          border-radius:2px; cursor:pointer; padding:0; }
  .step.on { background:var(--a); }
  .row { display:flex; align-items:center; gap:8px; margin:4px 0; }
</style>
</head>
<body>
<header>
  <div class="big">♪ <span id="hbeat">–</span></div>
  <div class="grp">速度 <input id="tempo" type="range" min="60" max="180" step="0.5">
    <span id="tempoVal">–</span> BPM</div>
  <div class="grp">
    <button id="play">播放</button><button id="stop">停止</button>
    <span class="muted" id="quantize"></span>
  </div>
  <div class="grp">主时钟
    <select id="master"></select>
    <span id="masterState" class="pill">–</span>
  </div>
  <div class="grp">
    <button id="monitor">启用监听声音</button>
    <span id="monState" class="muted"></span>
  </div>
</header>
<main>
  <div>
    <div class="card"><h3>共同节拍与各应用相位（乐句位置由共享节拍派生，相位一致 ≠ 小节一致）</h3>
      <canvas id="lanes" height="260"></canvas>
      <div id="stepEditors"></div>
    </div>
  </div>
  <div>
    <div class="card"><h3>跨端击打偏差（由合成输出实测）</h3>
      <table id="dev"><tr><th>应用对</th><th>n</th><th>均值</th><th>p99</th><th>最大</th></tr></table>
    </div>
    <div class="card" style="margin-top:12px"><h3>故障注入（真实 UDP 发送路径）</h3>
      <div class="row">丢包 <input id="loss" type="range" min="0" max="0.5" step="0.01" value="0"><span id="lossV">0%</span></div>
      <div class="row">抖动 <input id="jitter" type="range" min="0" max="0.1" step="0.005" value="0"><span id="jitterV">0ms</span></div>
      <div class="row">重排 <input id="reorder" type="range" min="0" max="0.5" step="0.01" value="0"><span id="reorderV">0%</span></div>
      <div class="row"><button id="applyFaults">应用</button>
        <button id="kill" class="warn">宕掉主时钟</button>
        <button id="revive">复活</button></div>
      <div class="muted" id="netstats"></div>
    </div>
    <div class="card" style="margin-top:12px"><h3>运行诊断（相位漂移 / 执行证据）</h3>
      <div class="row"><span class="muted">运行状态</span>
        <span id="diagRun" class="pill">–</span>
        <span class="muted">跨端节奏</span>
        <span id="diagCross" class="pill">–</span>
        <span class="muted">监听声音</span>
        <span id="diagMon" class="pill">–</span>
      </div>
      <div class="muted" id="diagStages"></div>
      <div id="diagAlerts" class="muted" style="font-size:12px;margin-top:4px"></div>
      <div class="row"><button id="exportEv">导出证据 JSON</button>
        <span class="muted" id="evState"></span></div>
    </div>
    <div class="card" style="margin-top:12px"><h3>内核事件 / 失败反馈</h3>
      <div id="log"></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let state = null;
const flashes = {};  // (app,beat,inst) -> expiry timestamp, kernel-driven hits

// ---------- SSE state ----------
const es = new EventSource('/events');
es.onmessage = ev => {
  state = JSON.parse(ev.data);
  const now = performance.now();
  for (const app of state.apps)
    for (const h of app.hits) flashes[app.id + h.beat + h.inst] = now + 180;
  render();
};
es.onerror = () => { $('masterState').textContent = '连接断开'; };

// ---------- controls ----------
async function ctl(action, args={}) {
  const r = await fetch('/control', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action, ...args})});
  const j = await r.json();
  if (!j.ok) log('控制失败: ' + (j.error || action), true);
  return j;
}
$('play').onclick = () => ctl('play');
$('stop').onclick = () => ctl('stop');
$('tempo').oninput = () => ctl('set_tempo', {bpm: parseFloat($('tempo').value)});
$('master').onchange = () => ctl('set_master', {node: $('master').value});
$('applyFaults').onclick = () => ctl('inject', {
  loss: parseFloat($('loss').value), jitter: parseFloat($('jitter').value),
  reorder: parseFloat($('reorder').value)});
$('kill').onclick = () => ctl('kill');
$('revive').onclick = () => ctl('revive');
$('exportEv').onclick = async () => {
  $('evState').textContent = '导出中…';
  try {
    const r = await fetch('/evidence');
    const blob = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'interbeat-evidence.json';
    a.click();
    $('evState').textContent = '已导出，可用 beattrace diagnose 离线重放';
  } catch (e) { $('evState').textContent = '导出失败'; }
};
['loss','jitter','reorder'].forEach(id => $(id).oninput = () => {
  const v = parseFloat($(id).value);
  $(id+'V').textContent = id==='jitter' ? (v*1000).toFixed(0)+'ms' : Math.round(v*100)+'%';
});

// ---------- rendering ----------
const COLORS = {A:'#4fc3f7', B:'#aed581', C:'#ffb74d', D:'#ce93d8'};
const INST_SHAPE = {kick:'●', snare:'■', hat:'▲', blip:'◆'};
function render() {
  if (!state) return;
  $('hbeat').textContent = state.playing ? state.beat.toFixed(2) : ('– ' + state.beat.toFixed(2));
  $('tempoVal').textContent = state.tempo.toFixed(1);
  if (document.activeElement !== $('tempo')) $('tempo').value = state.tempo;
  $('quantize').textContent = state.playing
      ? '' : `量化边界=${state.quantum}拍 下一边界=${state.next_boundary}`;
  $('masterState').textContent = state.master ? `term ${state.term}` : '选举中…';
  const sel = $('master');
  if (sel.options.length !== state.apps.length) {
    sel.innerHTML = state.apps.map(a => `<option>${a.id}</option>`).join('');
  }
  sel.value = state.master || state.apps[0].id;

  // deviation table
  const dev = $('dev');
  dev.innerHTML = '<tr><th>应用对</th><th>n</th><th>均值</th><th>p99</th><th>最大</th></tr>' +
    Object.entries(state.dev).map(([k, d]) =>
      `<tr><td>${k}</td><td>${d.n}</td><td>${d.mean_ms.toFixed(1)}ms</td>` +
      `<td>${d.p99_ms.toFixed(1)}ms</td><td>${d.max_ms.toFixed(1)}ms</td></tr>`).join('');
  $('netstats').textContent = state.netstats;

  // running diagnosis: cross-end rhythm vs monitor sound, with stages
  const dg = state.diag || {};
  const pill = (el, on, okText, badText) => {
    el.textContent = on === null || on === undefined ? '–' : (on ? badText : okText);
    el.className = 'pill ' + (on ? 'bad' : (on === false ? 'on' : ''));
  };
  $('diagRun').textContent = dg.error ? '诊断失败' : (state.playing ? '播放中' : '停止');
  $('diagRun').className = 'pill ' + (dg.error ? 'bad' : (state.playing ? 'on' : ''));
  pill($('diagCross'), dg.cross_end, '正常', '异常');
  pill($('diagMon'), dg.monitor, '正常', '异常');
  const stages = dg.stages ? Object.entries(dg.stages)
    .map(([s, n]) => `${s}:${n}`).join('  ') : '';
  $('diagStages').textContent = dg.error ? ('诊断错误: ' + dg.error)
    : (stages ? ('异常阶段  ' + stages) : '');
  $('diagAlerts').innerHTML = (dg.alerts || []).map(a =>
    `<div class="${a.severity === 'warn' ? 'bad' : ''}">${a.severity === 'warn' ? '!' : '·'} ${a.text}</div>`).join('');

  // event log
  $('log').innerHTML = state.log.map(l =>
    `<div class="${l.bad?'bad':''}">${l.t} ${l.text}</div>`).join('');
  $('log').scrollTop = $('log').scrollHeight;

  drawLanes();
  drawStepEditors();
}

function drawLanes() {
  const cv = $('lanes'), ctx = cv.getContext('2d');
  const W = cv.width = cv.clientWidth * devicePixelRatio;
  const H = cv.height = 260 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = cv.clientWidth, h = 260;
  ctx.clearRect(0, 0, w, h);
  const now = performance.now();

  // shared beat ruler: one bar = 4 beats
  const beat = state.beat, span = 8;
  ctx.fillStyle = '#8a919c'; ctx.font = '11px monospace';
  const x0 = 40, x1 = w - 10, px = b => x0 + (b - (beat - 1)) / span * (x1 - x0);
  for (let b = Math.ceil(beat - 1); b <= beat + span - 1; b++) {
    const x = px(b);
    ctx.strokeStyle = b % 4 === 0 ? '#4a5260' : '#2c313a';
    ctx.beginPath(); ctx.moveTo(x, 16); ctx.lineTo(x, 30); ctx.stroke();
    if (b % 4 === 0) ctx.fillText(`beat ${b}`, x - 20, 12);
  }
  // shared playhead
  ctx.strokeStyle = '#ef5350';
  ctx.beginPath(); ctx.moveTo(px(beat), 14); ctx.lineTo(px(beat), h - 10); ctx.stroke();

  // app lanes
  const laneH = (h - 40) / state.apps.length;
  state.apps.forEach((app, i) => {
    const y = 34 + i * laneH, mid = y + laneH / 2;
    ctx.fillStyle = COLORS[app.id] || '#d7dce2';
    ctx.font = 'bold 13px monospace';
    ctx.fillText(app.id, 8, mid + 4);
    ctx.font = '10px monospace'; ctx.fillStyle = '#8a919c';
    ctx.fillText(app.alive ? `小节 ${app.bar}  相位 ${app.phrase_pos.toFixed(2)}/${app.phrase_beats}` +
      (app.is_master ? '  [主时钟]' : (app.locked ? '' : '  [未锁定]'))
      : '离线', 26, y + 12);
    if (!app.alive) return;
    // phrase cells
    const pw = (x1 - x0) / span, ph = app.phrase_beats;
    const pos = app.phrase_pos;
    for (let bcell = 0; bcell < ph; bcell += 1) {
      const cx = x0 + ((bcell - pos + 1) / span) * (x1 - x0);
      ctx.strokeStyle = Math.floor(bcell) % 1 === 0 ? '#3a414d' : '#262b33';
      ctx.strokeRect(cx, y + 18, pw, laneH - 26);
    }
    // steps
    for (const s of app.steps) {
      const sx = x0 + ((s.beat - pos + 1) / span) * (x1 - x0);
      if (sx < x0 - 8 || sx > x1) continue;
      const key = app.id + (app.phrase_base + s.beat) + s.inst;
      const keyPrev = app.id + (app.phrase_base - ph + s.beat) + s.inst;
      const hot = (flashes[key] && flashes[key] > now) ||
                  (flashes[keyPrev] && flashes[keyPrev] > now);
      ctx.fillStyle = hot ? '#ffffff' : (COLORS[app.id] || '#d7dce2');
      ctx.font = `${hot ? 16 : 12}px monospace`;
      ctx.fillText(INST_SHAPE[s.inst] || '•', sx - 4, mid + 8);
    }
  });
}

// step toggle editors (kernel-controlled pattern edit)
function drawStepEditors() {
  const host = $('stepEditors');
  if (host.dataset.built !== '1') {
    host.dataset.built = '1';
    host.innerHTML = state.apps.map(a =>
      `<div class="row"><span style="width:20px;color:${COLORS[a.id]}">${a.id}</span>` +
      `<span class="steps" id="steps-${a.id}"></span>` +
      `<span class="muted">乐句 <select id="ph-${a.id}">` +
      [2,3,4,6,8,12].map(p => `<option>${p}</option>`).join('') +
      `</select> 拍</span></div>`).join('');
    state.apps.forEach(a => {
      $('ph-' + a.id).onchange = e => ctl('set_phrase', {app: a.id, beats: parseInt(e.target.value)});
    });
  }
  state.apps.forEach(a => {
    $('ph-' + a.id).value = a.phrase_beats;
    const host2 = $('steps-' + a.id);
    const n = a.phrase_beats * 2;  // 8th-note grid
    if (host2.children.length !== n) {
      host2.innerHTML = Array.from({length: n}, (_, i) =>
        `<button class="step" data-i="${i}"></button>`).join('');
      [...host2.children].forEach(btn => btn.onclick = () =>
        ctl('toggle_step', {app: a.id, cell: parseInt(btn.dataset.i)}));
    }
    const onCells = new Set(a.steps.filter(s => s.vel > 0).map(s => Math.round(s.beat * 2)));
    [...host2.children].forEach((btn, i) =>
      btn.classList.toggle('on', onCells.has(i)));
  });
}

// ---------- monitor audio: the kernel's actual rendered mix ----------
let audioOn = false;
$('monitor').onclick = async () => {
  if (audioOn) return;
  audioOn = true;
  $('monState').textContent = '缓冲中…';
  const ctx = new AudioContext({sampleRate: 44100});
  const workletSrc = `class P extends AudioWorkletProcessor {
    constructor(){ super(); this.q=[]; this.pos=0;
      this.port.onmessage=e=>this.q.push(e.data); }
    process(i,o){ const out=o[0][0];
      for(let k=0;k<out.length;k++){
        if(this.q.length && this.pos>=this.q[0].length){ this.q.shift(); this.pos=0; }
        out[k]=this.q.length?this.q[0][this.pos++]:0; }
      return true; } }
    registerProcessor('pcm', P);`;
  await ctx.audioWorklet.addModule(
    URL.createObjectURL(new Blob([workletSrc], {type:'application/javascript'})));
  const node = new AudioWorkletNode(ctx, 'pcm');
  node.connect(ctx.destination);
  const res = await fetch('/monitor');
  const reader = res.body.getReader();
  let acc = new Uint8Array(0), started = false, chunks = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const merged = new Uint8Array(acc.length + value.length);
    merged.set(acc); merged.set(value, acc.length); acc = merged;
    const n = acc.length - (acc.length % 2);
    const s16 = new Int16Array(acc.buffer, 0, n / 2);
    const f32 = new Float32Array(n / 2);
    for (let k = 0; k < f32.length; k++) f32[k] = s16[k] / 32768;
    node.port.postMessage(f32, [f32.buffer]);
    acc = acc.slice(n);
    if (!started && ++chunks > 20) { started = true; $('monState').textContent = '监听中 44.1kHz'; }
  }
};
function log(text, bad) { /* transient; kernel log arrives via SSE */ }
</script>
</body>
</html>
"""
