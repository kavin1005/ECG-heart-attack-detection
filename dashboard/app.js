/**
 * CardioPulse AI – Frontend Interactive Application Controller
 * =============================================================
 * Handles:
 *  - High-DPI ECG Canvas rendering with medical oscilloscope grid & glow
 *  - Animated sweep-line mode
 *  - REST API calls to /api/sample, /api/predict, /api/catalog
 *  - Interactive class filtering & sample navigation
 *  - Web Audio API heartbeat monitor sound synth
 *  - Custom 187-point signal testing
 */

// ── Application State ───────────────────────────────────────────────────────
const state = {
  currentIndex: 0,
  maxIndex: 21891,
  currentSignal: [],
  currentTrueLabel: 0,
  currentClassMeta: null,
  catalog: {},          // {0: [indices...], 1: [indices...], ...}
  catalogPointers: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
  isAnimating: false,
  sweepProgress: 0,
  animFrameId: null,
  audioEnabled: false,
  audioCtx: null,
  showPqst: true,
  latestResult: null,
};

// ── DOM Elements ────────────────────────────────────────────────────────────
const canvas            = document.getElementById("ecgCanvas");
const ctx               = canvas.getContext("2d");
const canvasWrapper     = document.getElementById("canvasWrapper");
const canvasTooltip     = document.getElementById("canvasTooltip");

const currentSampleTag  = document.getElementById("currentSampleTag");
const sampleSlider      = document.getElementById("sampleSlider");
const sampleInput       = document.getElementById("sampleInput");
const prevBtn           = document.getElementById("prevBtn");
const nextBtn           = document.getElementById("nextBtn");
const randomBtn         = document.getElementById("randomBtn");
const loadBtn           = document.getElementById("loadBtn");
const toggleAnimBtn     = document.getElementById("toggleAnimBtn");
const togglePqstBtn     = document.getElementById("togglePqstBtn");

const severityBadge     = document.getElementById("severityBadge");
const matchBadge        = document.getElementById("matchBadge");
const diagnosisName     = document.getElementById("diagnosisName");
const diagnosisSub      = document.getElementById("diagnosisSub");
const confPercentVal    = document.getElementById("confPercentVal");
const confProgressBar   = document.getElementById("confProgressBar");
const trueLabelVal      = document.getElementById("trueLabelVal");
const predLabelVal      = document.getElementById("predLabelVal");
const latencyBadge      = document.getElementById("latencyBadge");
const probListContainer = document.getElementById("probListContainer");
const classButtons      = document.querySelectorAll(".cat-pill");

const audioToggleBtn    = document.getElementById("audioToggleBtn");
const audioIconOn       = document.getElementById("audioIconOn");
const audioIconOff      = document.getElementById("audioIconOff");

const exportReportBtn   = document.getElementById("exportReportBtn");
const reportModal       = document.getElementById("reportModal");
const closeReportBtn    = document.getElementById("closeReportBtn");
const printNowBtn       = document.getElementById("printNowBtn");
const reportCanvas      = document.getElementById("reportCanvas");

const viewMetricsBtn    = document.getElementById("viewMetricsBtn");
const metricsModal      = document.getElementById("metricsModal");
const closeMetricsBtn   = document.getElementById("closeMetricsBtn");

const openCustomModalBtn= document.getElementById("openCustomModalBtn");
const customModal       = document.getElementById("customModal");
const closeCustomBtn    = document.getElementById("closeCustomBtn");
const customTextarea    = document.getElementById("customSignalTextarea");
const generateSynthBtn  = document.getElementById("generateSyntheticBtn");
const evaluateCustomBtn = document.getElementById("evaluateCustomBtn");


// ── High-DPI Canvas Resizing ───────────────────────────────────────────────
function setupCanvasDPI() {
  const rect = canvasWrapper.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;
  canvas.width  = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
}

window.addEventListener("resize", () => {
  setupCanvasDPI();
  renderECG();
});


// ── Medical ECG Grid & Signal Drawing ──────────────────────────────────────
function drawMedicalGrid(width, height) {
  ctx.save();
  ctx.clearRect(0, 0, width, height);

  // Background tint
  ctx.fillStyle = "#03070d";
  ctx.fillRect(0, 0, width, height);

  const minorStep = 15;
  const majorStep = minorStep * 5; // standard medical 5x5 block

  // Minor grid lines
  ctx.strokeStyle = "rgba(0, 240, 255, 0.035)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < width; x += minorStep) {
    ctx.moveTo(x, 0); ctx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += minorStep) {
    ctx.moveTo(0, y); ctx.lineTo(width, y);
  }
  ctx.stroke();

  // Major grid lines
  ctx.strokeStyle = "rgba(0, 240, 255, 0.09)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let x = 0; x < width; x += majorStep) {
    ctx.moveTo(x, 0); ctx.lineTo(x, height);
  }
  for (let y = 0; y < height; y += majorStep) {
    ctx.moveTo(0, y); ctx.lineTo(width, y);
  }
  ctx.stroke();

  // Baseline zero line
  const baselineY = height * 0.78;
  ctx.strokeStyle = "rgba(0, 240, 255, 0.18)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, baselineY);
  ctx.lineTo(width, baselineY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.restore();
}

function renderECG() {
  const rect = canvasWrapper.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  drawMedicalGrid(width, height);

  if (!state.currentSignal || state.currentSignal.length === 0) return;

  const points = state.currentSignal;
  const n = points.length;

  const paddingX = 40;
  const plotWidth = width - paddingX * 2;
  const plotHeight = height * 0.70;
  const baselineY = height * 0.82;

  // Determine line color from diagnosis severity
  let strokeColor = "#00f0ff";
  let glowColor   = "rgba(0, 240, 255, 0.4)";
  if (state.currentClassMeta) {
    strokeColor = state.currentClassMeta.color || "#00f0ff";
  }

  // Draw waveform
  ctx.save();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = strokeColor;
  ctx.shadowColor = strokeColor;
  ctx.shadowBlur = 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  const maxRenderIndex = state.isAnimating 
    ? Math.floor(state.sweepProgress * n) 
    : n;

  for (let i = 0; i < maxRenderIndex; i++) {
    const x = paddingX + (i / (n - 1)) * plotWidth;
    const y = baselineY - (points[i] * plotHeight);
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  // Area under curve fill
  if (!state.isAnimating && maxRenderIndex > 0) {
    ctx.lineTo(paddingX + plotWidth, baselineY);
    ctx.lineTo(paddingX, baselineY);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, baselineY - plotHeight, 0, baselineY);
    grad.addColorStop(0, glowColor);
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.shadowBlur = 0;
    ctx.fill();
  }

  // Draw sweep beam if animating
  if (state.isAnimating && maxRenderIndex > 0 && maxRenderIndex < n) {
    const sweepX = paddingX + (maxRenderIndex / (n - 1)) * plotWidth;
    ctx.save();
    ctx.shadowBlur = 15;
    ctx.shadowColor = "#fff";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sweepX, 10);
    ctx.lineTo(sweepX, height - 10);
    ctx.stroke();
    ctx.restore();
  }

  // Draw P-Q-R-S-T wave markers if enabled and not currently sweeping
  if (state.showPqst && !state.isAnimating) {
    drawPQSTMarkers(points, paddingX, plotWidth, plotHeight, baselineY);
  }

  ctx.restore();
}

// ── Automated P-Q-R-S-T Peak Detection ──────────────────────────────────────
function detectPQST(points) {
  if (!points || points.length < 50) return null;

  // 1. R-Peak: Global maximum within realistic QRS window
  let rIdx = 0;
  let maxR = -Infinity;
  const rSearchEnd = Math.min(points.length - 20, 130);
  for (let i = 10; i < rSearchEnd; i++) {
    if (points[i] > maxR) {
      maxR = points[i];
      rIdx = i;
    }
  }

  // 2. Q-Valley: Lowest valley before R
  let qIdx = Math.max(0, rIdx - 1);
  let minQ = points[qIdx];
  const qStart = Math.max(0, rIdx - 25);
  for (let i = qStart; i < rIdx; i++) {
    if (points[i] < minQ) {
      minQ = points[i];
      qIdx = i;
    }
  }

  // 3. S-Valley: Lowest valley after R
  let sIdx = Math.min(points.length - 1, rIdx + 1);
  let minS = points[sIdx];
  const sEnd = Math.min(points.length - 1, rIdx + 25);
  for (let i = rIdx + 1; i <= sEnd; i++) {
    if (points[i] < minS) {
      minS = points[i];
      sIdx = i;
    }
  }

  // 4. P-Peak: Preceding atrial depolarization
  let pIdx = Math.max(0, Math.floor(qIdx / 2));
  let maxP = -Infinity;
  for (let i = Math.max(0, qIdx - 40); i < qIdx; i++) {
    if (points[i] > maxP) {
      maxP = points[i];
      pIdx = i;
    }
  }

  // 5. T-Peak: Following ventricular repolarization dome
  let tIdx = Math.min(points.length - 1, sIdx + 25);
  let maxT = -Infinity;
  const tEnd = Math.min(points.length - 1, sIdx + 70);
  for (let i = sIdx + 8; i <= tEnd; i++) {
    if (points[i] > maxT) {
      maxT = points[i];
      tIdx = i;
    }
  }

  return {
    P: { index: pIdx, val: points[pIdx], color: "#ffb300" },
    Q: { index: qIdx, val: points[qIdx], color: "#00e676" },
    R: { index: rIdx, val: points[rIdx], color: "#ff1744" },
    S: { index: sIdx, val: points[sIdx], color: "#00e676" },
    T: { index: tIdx, val: points[tIdx], color: "#00f0ff" },
  };
}

function drawPQSTMarkers(points, paddingX, plotWidth, plotHeight, baselineY) {
  const peaks = detectPQST(points);
  if (!peaks) return;

  const n = points.length;

  for (const [key, item] of Object.entries(peaks)) {
    const x = paddingX + (item.index / (n - 1)) * plotWidth;
    const y = baselineY - (item.val * plotHeight);

    ctx.save();
    // Dashed guide line
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x, y - 28);
    ctx.stroke();
    ctx.setLineDash([]);

    // Anchor dot on curve
    ctx.fillStyle = item.color;
    ctx.shadowColor = item.color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Labeled badge pin
    const badgeY = y - 36;
    ctx.fillStyle = "rgba(14, 20, 36, 0.9)";
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(x - 9, badgeY - 8, 18, 16, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(key, x, badgeY);

    ctx.restore();
  }
}


// ── Sweep-Line Animation Loop ──────────────────────────────────────────────
function animateSweep() {
  if (!state.isAnimating) return;

  state.sweepProgress += 0.012; // speed of beam
  if (state.sweepProgress > 1.0) {
    state.sweepProgress = 0;
    playHeartbeatSound();
  }

  renderECG();
  state.animFrameId = requestAnimationFrame(animateSweep);
}

function toggleAnimation() {
  state.isAnimating = !state.isAnimating;
  if (state.isAnimating) {
    toggleAnimBtn.classList.add("active");
    toggleAnimBtn.style.color = "var(--ecg-cyan)";
    state.sweepProgress = 0;
    animateSweep();
  } else {
    toggleAnimBtn.classList.remove("active");
    toggleAnimBtn.style.color = "";
    cancelAnimationFrame(state.animFrameId);
    state.sweepProgress = 1.0;
    renderECG();
  }
}


// ── Audio Heartbeat Synthesizer (Web Audio API) ─────────────────────────────
function initAudio() {
  if (!state.audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    state.audioCtx = new AudioContext();
  }
  if (state.audioCtx.state === "suspended") {
    state.audioCtx.resume();
  }
}

function playHeartbeatSound() {
  if (!state.audioEnabled) return;
  try {
    initAudio();
    const ctx = state.audioCtx;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    // Higher beep for abnormal beat, standard beep for normal
    const freq = (state.currentTrueLabel === 0) ? 580 : 820;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(120, ctx.currentTime + 0.08);

    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.09);
  } catch (e) {
    console.warn("Audio playback error:", e);
  }
}

audioToggleBtn.addEventListener("click", () => {
  initAudio();
  state.audioEnabled = !state.audioEnabled;
  if (state.audioEnabled) {
    audioIconOn.classList.remove("hidden");
    audioIconOff.classList.add("hidden");
    audioToggleBtn.style.borderColor = "var(--ecg-cyan)";
    playHeartbeatSound();
  } else {
    audioIconOn.classList.add("hidden");
    audioIconOff.classList.remove("hidden");
    audioToggleBtn.style.borderColor = "";
  }
});


// ── Tooltip Inspection on Canvas ───────────────────────────────────────────
canvasWrapper.addEventListener("mousemove", (e) => {
  if (!state.currentSignal || state.currentSignal.length === 0) return;
  const rect = canvasWrapper.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const paddingX = 40;
  const plotWidth = rect.width - paddingX * 2;

  if (mouseX >= paddingX && mouseX <= paddingX + plotWidth) {
    const ratio = (mouseX - paddingX) / plotWidth;
    const ptIndex = Math.min(
      Math.max(0, Math.floor(ratio * state.currentSignal.length)),
      state.currentSignal.length - 1
    );
    const amp = state.currentSignal[ptIndex];

    canvasTooltip.style.left = `${mouseX}px`;
    canvasTooltip.style.top  = `${rect.height * 0.25}px`;
    canvasTooltip.innerHTML  = `<strong>t = ${ptIndex}</strong> (amp: ${amp.toFixed(4)})`;
    canvasTooltip.classList.remove("hidden");
  } else {
    canvasTooltip.classList.add("hidden");
  }
});

canvasWrapper.addEventListener("mouseleave", () => {
  canvasTooltip.classList.add("hidden");
});


// ── REST API Communications ────────────────────────────────────────────────
async function loadSample(index) {
  index = Math.max(0, Math.min(index, state.maxIndex));
  state.currentIndex = index;

  // Update inputs
  sampleSlider.value = index;
  sampleInput.value  = index;
  currentSampleTag.textContent = `Sample #${index}`;

  try {
    // 1. Fetch signal points
    const res = await fetch(`/api/sample?index=${index}`);
    if (!res.ok) throw new Error("Failed to load sample data");
    const data = await res.json();

    state.currentSignal    = data.signal;
    state.currentTrueLabel = data.true_label;
    state.currentClassMeta = data.class_meta;

    // 2. Trigger CNN model inference
    await runInference(index);

    // 3. Render waveform
    renderECG();
    playHeartbeatSound();

  } catch (err) {
    console.error("Error loading sample:", err);
  }
}

async function runInference(index) {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: index })
    });
    if (!res.ok) throw new Error("Inference call failed");
    const result = await res.json();

    updateDiagnosisUI(result);
  } catch (err) {
    console.error("Inference error:", err);
  }
}

async function runCustomInference(signalArray) {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ signal: signalArray })
    });
    if (!res.ok) throw new Error("Custom inference failed");
    const result = await res.json();

    state.currentSignal = signalArray;
    state.currentTrueLabel = -1;
    state.currentClassMeta = result.class_meta;
    currentSampleTag.textContent = "Custom Signal";

    updateDiagnosisUI(result, true);
    renderECG();
  } catch (err) {
    alert("Error evaluating custom signal: " + err.message);
  }
}


// ── UI Updates from Model Inference ────────────────────────────────────────
function updateDiagnosisUI(result, isCustom = false) {
  state.latestResult = result;
  // Latency
  latencyBadge.textContent = `⚡ ${result.inference_time_ms} ms`;

  // Triage severity badge
  const meta = result.class_meta;
  severityBadge.className = `triage-badge ${meta.severity}`;
  severityBadge.textContent = `${meta.severity.toUpperCase()} RISK`;

  // Match / Mismatch badge
  if (isCustom) {
    matchBadge.className = "match-badge";
    matchBadge.textContent = "CUSTOM INPUT EVALUATION";
  } else if (result.is_correct) {
    matchBadge.className = "match-badge match";
    matchBadge.textContent = "✓ GROUND TRUTH MATCH";
  } else {
    matchBadge.className = "match-badge mismatch";
    matchBadge.textContent = "✗ PREDICTION MISMATCH";
  }

  // Diagnosis title & description
  diagnosisName.textContent = meta.name;
  diagnosisSub.textContent  = meta.description;

  // Confidence meter
  confPercentVal.textContent  = `${result.confidence}%`;
  confProgressBar.style.width = `${result.confidence}%`;

  // Comparison row
  if (isCustom) {
    trueLabelVal.textContent = "N/A (User Supplied)";
  } else {
    const trueName = result.true_class_meta ? result.true_class_meta.name : `Class ${result.true_label}`;
    trueLabelVal.textContent = `${result.true_label} · ${trueName}`;
  }
  predLabelVal.textContent = `${result.predicted_label} · ${meta.name}`;

  // Probability bars
  renderProbabilityBars(result.probabilities, result.predicted_label);
}

function renderProbabilityBars(probs, predictedId) {
  probListContainer.innerHTML = "";

  probs.forEach(item => {
    const row = document.createElement("div");
    row.className = `prob-row ${item.class_id === predictedId ? "active" : ""}`;

    row.innerHTML = `
      <div class="prob-header">
        <span class="prob-name" style="color: ${item.color}">${item.class_id} · ${item.name}</span>
        <span class="prob-val">${item.prob.toFixed(1)}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width: ${item.prob}%; background: ${item.color};"></div>
      </div>
    `;
    probListContainer.appendChild(row);
  });
}


// ── Catalog & Quick Class Explorers ────────────────────────────────────────
async function loadCatalog() {
  try {
    const res = await fetch("/api/catalog");
    if (res.ok) {
      const data = await res.json();
      state.catalog = data.catalog;
    }
  } catch (e) {
    console.warn("Could not load catalog:", e);
  }
}

classButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const clsId = parseInt(btn.dataset.class);
    classButtons.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    // Jump to next sample in catalog for this class
    if (state.catalog[clsId] && state.catalog[clsId].length > 0) {
      const list = state.catalog[clsId];
      const ptr = state.catalogPointers[clsId] % list.length;
      state.catalogPointers[clsId] = ptr + 1;
      const targetIndex = list[ptr];
      loadSample(targetIndex);
    }
  });
});


// ── Event Handlers & Navigation ────────────────────────────────────────────
prevBtn.addEventListener("click", () => {
  loadSample(state.currentIndex - 1);
});

nextBtn.addEventListener("click", () => {
  loadSample(state.currentIndex + 1);
});

randomBtn.addEventListener("click", () => {
  const randIdx = Math.floor(Math.random() * (state.maxIndex + 1));
  loadSample(randIdx);
});

loadBtn.addEventListener("click", () => {
  const val = parseInt(sampleInput.value);
  if (!isNaN(val)) loadSample(val);
});

sampleSlider.addEventListener("input", (e) => {
  sampleInput.value = e.target.value;
});

sampleSlider.addEventListener("change", (e) => {
  loadSample(parseInt(e.target.value));
});

sampleInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    loadSample(parseInt(sampleInput.value));
  }
});

toggleAnimBtn.addEventListener("click", toggleAnimation);

togglePqstBtn.addEventListener("click", () => {
  state.showPqst = !state.showPqst;
  togglePqstBtn.classList.toggle("active", state.showPqst);
  renderECG();
});

// ── Medical Diagnostic Report Logic ───────────────────────────────────────
function openReportModal() {
  if (!state.currentSignal || state.currentSignal.length === 0) return;
  const res = state.latestResult;
  if (!res) return;

  const now = new Date();
  document.getElementById("reportDate").textContent = now.toLocaleDateString() + " " + now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  document.getElementById("reportRecordId").textContent = (state.currentTrueLabel >= 0) ? `MITBIH-TEST-#${state.currentIndex}` : "CUSTOM-ECG-UPLOAD";

  const findingName = document.getElementById("reportFindingName");
  findingName.textContent = res.class_meta.name;
  
  const triageBlock = document.getElementById("reportTriageBlock");
  if (res.class_meta.severity === "high") {
    triageBlock.className = "report-triage-block high-risk";
  } else {
    triageBlock.className = "report-triage-block";
  }

  document.getElementById("reportTriageSeverity").textContent = `${res.class_meta.severity.toUpperCase()} RISK`;
  document.getElementById("reportConfVal").textContent = `${res.confidence}%`;
  document.getElementById("reportTruthVal").textContent = (state.currentTrueLabel >= 0) 
    ? (res.true_class_meta ? res.true_class_meta.name : `Class ${state.currentTrueLabel}`)
    : "Custom Input";

  // Impression text
  const imp = document.getElementById("reportImpressionText");
  if (res.predicted_label === 0) {
    imp.textContent = "Sinus rhythm within normal limits. Consistent R-R intervals with sharp QRS complex. No acute ventricular or supraventricular ectopy detected on this lead.";
  } else if (res.predicted_label === 2) {
    imp.textContent = "CRITICAL: Premature Ventricular Contraction (PVC) / Ventricular Ectopy pattern detected. Distinct widened QRS complex with discordant T-wave repolarization. Clinical evaluation recommended.";
  } else if (res.predicted_label === 1) {
    imp.textContent = "Supraventricular ectopic activity identified. Atrial premature depolarization with narrow ventricular response. Monitor patient for paroxysmal atrial tachyarrhythmias.";
  } else if (res.predicted_label === 3) {
    imp.textContent = "Fusion heartbeat identified. Combined simultaneous depolarization from supraventricular pacemaker and ventricular focus. Follow-up 12-lead ECG indicated.";
  } else {
    imp.textContent = "Waveform exhibits pacing spike artifacts or high distortion unclassifiable by standard morphological rhythm rules.";
  }

  // Populate probability grid
  const grid = document.getElementById("reportProbGrid");
  grid.innerHTML = "";
  res.probabilities.forEach(p => {
    const card = document.createElement("div");
    card.className = `report-prob-card ${p.class_id === res.predicted_label ? "pred-top" : ""}`;
    card.innerHTML = `
      <div class="report-p-name">${p.name}</div>
      <div class="report-p-val">${p.prob.toFixed(1)}%</div>
    `;
    grid.appendChild(card);
  });

  // Render static high-contrast ECG on report canvas
  renderReportCanvas();

  reportModal.classList.remove("hidden");
}

function renderReportCanvas() {
  const rCanvas = document.getElementById("reportCanvas");
  if (!rCanvas) return;
  const rCtx = rCanvas.getContext("2d");
  const w = rCanvas.width;
  const h = rCanvas.height;

  // Clean clinical background with millimeter grid
  rCtx.fillStyle = "#ffffff";
  rCtx.fillRect(0, 0, w, h);

  // Grid lines
  rCtx.strokeStyle = "#e8eef5";
  rCtx.lineWidth = 1;
  rCtx.beginPath();
  for (let x = 0; x < w; x += 15) { rCtx.moveTo(x, 0); rCtx.lineTo(x, h); }
  for (let y = 0; y < h; y += 15) { rCtx.moveTo(0, y); rCtx.lineTo(w, y); }
  rCtx.stroke();

  rCtx.strokeStyle = "#c8d8ec";
  rCtx.lineWidth = 1.2;
  rCtx.beginPath();
  for (let x = 0; x < w; x += 75) { rCtx.moveTo(x, 0); rCtx.lineTo(x, h); }
  for (let y = 0; y < h; y += 75) { rCtx.moveTo(0, y); rCtx.lineTo(w, y); }
  rCtx.stroke();

  // Baseline
  const baseY = h * 0.82;
  rCtx.strokeStyle = "#99b";
  rCtx.setLineDash([3, 3]);
  rCtx.beginPath();
  rCtx.moveTo(0, baseY);
  rCtx.lineTo(w, baseY);
  rCtx.stroke();
  rCtx.setLineDash([]);

  // Plot waveform
  const pts = state.currentSignal;
  const n = pts.length;
  const padX = 40;
  const plotW = w - padX * 2;
  const plotH = h * 0.70;

  rCtx.strokeStyle = "#003366";
  rCtx.lineWidth = 2.2;
  rCtx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = padX + (i / (n - 1)) * plotW;
    const y = baseY - (pts[i] * plotH);
    if (i === 0) rCtx.moveTo(x, y);
    else rCtx.lineTo(x, y);
  }
  rCtx.stroke();

  // Annotate P-Q-R-S-T peaks on report
  const peaks = detectPQST(pts);
  if (peaks) {
    for (const [key, item] of Object.entries(peaks)) {
      const x = padX + (item.index / (n - 1)) * plotW;
      const y = baseY - (item.val * plotH);

      rCtx.fillStyle = "#d32f2f";
      rCtx.beginPath();
      rCtx.arc(x, y, 3, 0, Math.PI * 2);
      rCtx.fill();

      rCtx.fillStyle = "#111";
      rCtx.font = "bold 11px sans-serif";
      rCtx.textAlign = "center";
      rCtx.fillText(key, x, y - 8);
    }
  }
}

exportReportBtn.addEventListener("click", openReportModal);
closeReportBtn.addEventListener("click", () => reportModal.classList.add("hidden"));
printNowBtn.addEventListener("click", () => window.print());
reportModal.addEventListener("click", (e) => {
  if (e.target === reportModal) reportModal.classList.add("hidden");
});


// ── Modals Logic ───────────────────────────────────────────────────────────
viewMetricsBtn.addEventListener("click", () => metricsModal.classList.remove("hidden"));
closeMetricsBtn.addEventListener("click", () => metricsModal.classList.add("hidden"));
metricsModal.addEventListener("click", (e) => {
  if (e.target === metricsModal) metricsModal.classList.add("hidden");
});

openCustomModalBtn.addEventListener("click", () => customModal.classList.remove("hidden"));
closeCustomBtn.addEventListener("click", () => customModal.classList.add("hidden"));
customModal.addEventListener("click", (e) => {
  if (e.target === customModal) customModal.classList.add("hidden");
});

// Synthetic Waveform Generator
generateSynthBtn.addEventListener("click", () => {
  // Generate classic 187-pt normal sinus beat with P-Q-R-S-T complexes
  const pts = [];
  for (let i = 0; i < 187; i++) {
    const t = i / 187;
    // Isoelectric baseline
    let val = 0.15;
    // P wave
    if (t >= 0.12 && t <= 0.22) {
      val += 0.15 * Math.sin(((t - 0.12) / 0.10) * Math.PI);
    }
    // Q drop
    if (t >= 0.28 && t <= 0.31) {
      val -= 0.10 * Math.sin(((t - 0.28) / 0.03) * Math.PI);
    }
    // R spike
    if (t >= 0.31 && t <= 0.38) {
      val += 0.80 * Math.sin(((t - 0.31) / 0.07) * Math.PI);
    }
    // S drop
    if (t >= 0.38 && t <= 0.42) {
      val -= 0.20 * Math.sin(((t - 0.38) / 0.04) * Math.PI);
    }
    // T wave
    if (t >= 0.52 && t <= 0.72) {
      val += 0.28 * Math.sin(((t - 0.52) / 0.20) * Math.PI);
    }
    pts.push(Math.max(0, Math.min(1, val)).toFixed(4));
  }
  customTextarea.value = pts.join(", ");
});

evaluateCustomBtn.addEventListener("click", () => {
  const raw = customTextarea.value.trim();
  if (!raw) return alert("Please enter 187 comma-separated numbers.");
  const nums = raw.split(/[\s,]+/).map(Number).filter(n => !isNaN(n));
  if (nums.length !== 187) {
    return alert(`Expected exactly 187 values, but found ${nums.length}.`);
  }
  customModal.classList.add("hidden");
  runCustomInference(nums);
});


// ── Startup Initialization ─────────────────────────────────────────────────
async function init() {
  setupCanvasDPI();
  await loadCatalog();
  await loadSample(0);
}

document.addEventListener("DOMContentLoaded", init);
