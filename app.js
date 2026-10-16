/* Open Hair Stack — Precision Client-Side Follicular-Unit Analyzer.
   Model runs 100% locally in the browser via onnxruntime-web. No uploads, no server.

   Pipeline: letterbox to 640x640 -> ONNX session -> transpose output ->
   class scores (sigmoid-activated in exported graph) -> class-agnostic NMS ->
   map back to image coordinates -> draw, count, band-split, heuristic, plan simulation.
*/

const MODEL_URL = "https://huggingface.co/openhairstack/follicle-fu-v0.1/resolve/main/model.onnx";
const CLASSES = ["fu1", "fu2", "fu3", "fu4", "fu5"];
const HAIRS = { fu1: 1, fu2: 2, fu3: 3, fu4: 4, fu5: 5 };
const COLORS = {
  fu1: "#e6194b", // 1 hair: vibrant red
  fu2: "#f78230", // 2 hairs: clinical orange
  fu3: "#3cb44b", // 3 hairs: emerald green
  fu4: "#0082c8", // 4 hairs: sapphire blue
  fu5: "#94a3b8"  // 5+ hairs: slate gray
};
const TARGET = 25, DONOR_AREA = 150, HARVEST = 0.3, ZONE_AREAS = [45, 45, 40];
const ZONE_NAMES = ["anterior", "middle", "posterior"];

// Broad browser compatibility for static Hugging Face Space iframes
if (typeof ort !== "undefined" && ort.env && ort.env.wasm) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
}

const $ = (id) => document.getElementById(id);
const statusBeacon = $("statusBeacon");
const statusText = $("statusText");

let session = null;
let img = null;
let lastResult = null;
let cachedInference = null;
let currentDets = [];
let currentPhotoName = "scalp-photo.jpg";
let activeClassFilter = "all";

const setStatus = (msg, state = "info") => {
  statusText.textContent = msg;
  statusBeacon.className = "beacon";
  if (state === "ready" || state === "ok") {
    statusBeacon.classList.add("beacon-green");
  } else if (state === "loading" || state === "pulse") {
    statusBeacon.classList.add("beacon-pulse");
  } else if (state === "warn") {
    statusBeacon.style.background = "#f59e0b";
    statusBeacon.style.boxShadow = "0 0 8px #f59e0b";
  } else if (state === "error") {
    statusBeacon.style.background = "#ef4444";
    statusBeacon.style.boxShadow = "0 0 8px #ef4444";
  }
};

/* ---------- Model Loading ---------- */
async function loadModel() {
  setStatus("Downloading follicle-fu-v0.1 neural weights (~10.6 MB)…", "pulse");
  try {
    const resp = await fetch(MODEL_URL, { method: "HEAD" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  } catch (e) {
    setStatus("Model weights unreachable (" + e.message + "). Operating in preview mode.", "warn");
    return;
  }

  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    $("analyze").disabled = false;
    setStatus("Ready: follicle-fu-v0.1 loaded (YOLO11n ONNX) · WebAssembly + SIMD", "ready");
    if (img && !lastResult) {
      analyze(true);
    }
  } catch (e) {
    console.error("ONNX initialization error:", e);
    setStatus("Model initialization failed: " + e.message, "error");
  }
}

/* ---------- Image Helper ---------- */
function loadImageFromSource(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not load image"));
    im.src = src;
  });
}

/* ---------- Preprocess + Inference ---------- */
function letterbox(image) {
  const S = 640;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = S;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, S, S);

  const scale = Math.min(S / image.width, S / image.height);
  const w = image.width * scale, h = image.height * scale;
  const padX = (S - w) / 2, padY = (S - h) / 2;
  ctx.drawImage(image, padX, padY, w, h);

  const data = ctx.getImageData(0, 0, S, S).data;
  const planes = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    planes[i] = data[i * 4] / 255;                 // R
    planes[S * S + i] = data[i * 4 + 1] / 255;     // G
    planes[2 * S * S + i] = data[i * 4 + 2] / 255; // B
  }
  return { planes, scale, padX, padY };
}

function transposeAndDecode(output) {
  const dims = output.dims;
  const nc = CLASSES.length;
  let boxes = [];

  if (dims[1] === 4 + nc) { // rows layout: [1, 9, 8400]
    const N = dims[2], d = output.data;
    for (let j = 0; j < N; j++) {
      const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
      let best = -1, bestCls = 0;
      for (let c = 0; c < nc; c++) {
        const s = d[(4 + c) * N + j];
        if (s > best) { best = s; bestCls = c; }
      }
      if (best >= 0.005) boxes.push({ cx, cy, w, h, score: best, cls: bestCls });
    }
  } else { // columns layout: [1, 8400, 9]
    const N = dims[1], stride = dims[2];
    for (let i = 0; i < N; i++) {
      const base = i * stride;
      let best = -1, bestCls = 0;
      for (let c = 0; c < nc; c++) {
        const s = output.data[base + 4 + c];
        if (s > best) { best = s; bestCls = c; }
      }
      if (best >= 0.005) boxes.push({
        cx: output.data[base], cy: output.data[base + 1],
        w: output.data[base + 2], h: output.data[base + 3], score: best, cls: bestCls,
      });
    }
  }
  return boxes;
}

function iou(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2, ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2, bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const x1 = Math.max(ax1, bx1), y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2), y2 = Math.min(ay2, by2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

function nms(boxes, iouThresh) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  while (boxes.length) {
    const b = boxes.shift();
    keep.push(b);
    boxes = boxes.filter((o) => !(iou(o, b) > iouThresh));
  }
  return keep;
}

/* ---------- Stats & Clinical Summarization ---------- */
function summarize(dets, image, pxmm) {
  const counts = {}; CLASSES.forEach(c => counts[c] = 0);
  dets.forEach(d => counts[CLASSES[d.cls]]++);
  const total = dets.length;
  const mean = total ? dets.reduce((s, d) => s + HAIRS[CLASSES[d.cls]], 0) / total : 0;
  const known = pxmm && pxmm >= 1;
  const w = image.width, h = image.height;
  const areaCm2 = known ? (w * h) / (pxmm * pxmm) / 100 : null;
  const density = areaCm2 ? total / areaCm2 : null;

  const bands = ZONE_NAMES.map((name, i) => {
    const y0 = h * i / 3, y1 = h * (i + 1) / 3;
    const inBand = dets.filter(d => d.cy >= y0 && d.cy < y1);
    const area = known ? (w * (y1 - y0)) / (pxmm * pxmm) / 100 : null;
    const fu1Share = inBand.length ? inBand.filter(d => d.cls === 0).length / inBand.length : 0;
    return { name, count: inBand.length, area, density: area ? inBand.length / area : null, fu1Share };
  });

  let norwoodTitle = "Inconclusive";
  let norwoodSub = "No units detected — try lowering confidence threshold or use a clear macro photo";
  let norwoodSeverity = "neutral";

  if (total > 0) {
    if (bands[2].count > 0 && bands[0].count === 0) {
      norwoodTitle = "NW ~3–4 (Frontal Thinning)";
      norwoodSub = "Marked anterior deficit with preserved posterior density gradient";
      norwoodSeverity = "warn";
    } else if (bands[2].count > 0 && bands[0].count > 0) {
      const ratio = (bands[0].fu1Share + 0.05) / (bands[2].fu1Share + 0.05);
      if (ratio > 1.6 || bands[0].count < bands[2].count * 0.6) {
        norwoodTitle = "NW ~3–4 (Frontal Thinning)";
        norwoodSub = "Significant anterior miniaturization / thinning gradient detected";
        norwoodSeverity = "warn";
      } else if (ratio > 1.25 || bands[0].count < bands[2].count * 0.85) {
        norwoodTitle = "NW ~2 (Mild Receding)";
        norwoodSub = "Mild anterior/posterior density gradient detected";
        norwoodSeverity = "info";
      } else {
        norwoodTitle = "NW 1 (Uniform Distribution)";
        norwoodSub = "No significant anterior/posterior thinning gradient across visible crop";
        norwoodSeverity = "good";
      }
    } else {
      norwoodTitle = "Localized Crop";
      norwoodSub = "Follicular units detected; uniform distribution across visible field";
      norwoodSeverity = "info";
    }
  }

  const capacity = density ? Math.round(density * DONOR_AREA * HARVEST) : 1125;
  const supplyNote = density
    ? `Measured donor capacity est. ${capacity} FU (measured density ${density.toFixed(1)} FU/cm²)`
    : "Assumed donor capacity ~25 FU/cm² × 150 cm² × 30% safe harvest";

  const rows = bands.map((b, i) => {
    const area = b.area || ZONE_AREAS[i];
    const zd = b.density || 20;
    const deficit = Math.max(0, TARGET - zd);
    return {
      zone: b.name[0].toUpperCase() + b.name.slice(1),
      area,
      zd: b.density,
      deficit,
      target: Math.round(area * deficit),
      allocated: 0
    };
  });

  const demand = rows.reduce((s, r) => s + r.target, 0);
  const scale = demand > 0 ? Math.min(1, capacity / demand) : 1;
  rows.forEach(r => r.allocated = Math.round(r.target * scale));

  return {
    counts, total, mean, areaCm2, density, bands,
    norwoodTitle, norwoodSub, norwoodSeverity,
    rows, capacity, supplyNote, scale
  };
}

/* ---------- Canvas Drawing & HUD Overlay ---------- */
function draw(canvas, image, dets) {
  const ctx = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);

  // Filter detections if a specific chip is selected
  const visibleDets = activeClassFilter === "all"
    ? dets
    : dets.filter(d => CLASSES[d.cls] === activeClassFilter);

  // Bounding boxes
  visibleDets.forEach(d => {
    const cls = CLASSES[d.cls];
    ctx.strokeStyle = COLORS[cls];
    ctx.lineWidth = Math.max(2, Math.round(image.width / 320));
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

    // Label tag
    ctx.fillStyle = COLORS[cls];
    const fontSize = Math.max(11, Math.round(image.width / 50));
    ctx.font = `bold ${fontSize}px var(--font), sans-serif`;
    ctx.fillText(`${cls} ${d.score.toFixed(2)}`, d.x1, Math.max(fontSize + 2, d.y1 - 4));
  });

  // Dark HUD Box in top-left
  const hudW = Math.max(130, Math.round(image.width * 0.22));
  const hudH = CLASSES.length * 22 + 14;
  ctx.fillStyle = "rgba(7, 10, 19, 0.82)";
  ctx.strokeStyle = "rgba(30, 46, 74, 0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(10, 10, hudW, hudH, 6);
  else ctx.rect(10, 10, hudW, hudH);
  ctx.fill();
  ctx.stroke();

  let y = 28;
  const labelFontSize = Math.max(12, Math.round(image.width / 52));
  ctx.font = `500 ${labelFontSize}px var(--mono), sans-serif`;
  CLASSES.forEach(c => {
    ctx.fillStyle = COLORS[c];
    ctx.fillRect(18, y - 9, 10, 10);
    ctx.fillStyle = "#e2e8f0";
    ctx.fillText(`${c}: ${lastResult.counts[c]}`, 36, y);
    y += 22;
  });
}

/* ---------- Chart Rendering ---------- */
function drawChart(cv, rows) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#070a13";
  ctx.fillRect(0, 0, W, H);

  const max = Math.max(...rows.map(r => r.target), 1);
  rows.forEach((r, i) => {
    const y = 24 + i * 48;
    const barW = W - 190;

    // Label
    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText(r.zone, 12, y + 14);

    // Track outlines
    ctx.strokeStyle = "#1e2e4a";
    ctx.strokeRect(95, y, barW, 11);
    ctx.strokeRect(95, y + 15, barW, 11);

    // Green = Allocated Grafts
    ctx.fillStyle = "#10b981";
    ctx.fillRect(95, y, barW * (r.allocated / max), 11);

    // Cyan = Target Demand
    ctx.fillStyle = "#00e5ff";
    ctx.fillRect(95, y + 15, barW * (r.target / max), 11);

    // Number text
    ctx.fillStyle = "#f8fafc";
    ctx.font = "12px monospace";
    ctx.fillText(`${r.allocated} / ${r.target}`, 95 + barW + 10, y + 14);
  });

  ctx.fillStyle = "#64748b";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillText("■ green: allocated grafts   ■ cyan: target demand", 95, H - 10);
}

/* ---------- Analysis Execution ---------- */
async function runInferenceOnImage(image) {
  const { planes, scale, padX, padY } = letterbox(image);
  const input = new ort.Tensor("float32", planes, [1, 3, 640, 640]);
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  const rawBoxes = transposeAndDecode(out[session.outputNames[0]]);
  return { rawBoxes, scale, padX, padY };
}

async function analyze(forceNewInference = true) {
  if (!img) return;
  const conf = parseFloat($("conf").value);
  const pxmm = parseFloat($("pxmm").value) || null;
  let dets = [];

  const spinner = $("analyzeSpinner");
  const btnText = $("analyzeBtnText");
  spinner.classList.remove("hidden");
  btnText.textContent = "Analyzing…";

  if (session) {
    if (forceNewInference || !cachedInference) {
      setStatus("Running neural inference in WebAssembly runtime…", "pulse");
      cachedInference = await runInferenceOnImage(img);
      setStatus("Ready: follicle-fu-v0.1 loaded · WebAssembly + SIMD", "ready");
    }
    const { rawBoxes, scale, padX, padY } = cachedInference;
    const candidates = rawBoxes.filter(b => b.score >= conf);
    const kept = nms(candidates, 0.45).slice(0, 400);

    dets = kept.map(b => {
      const cx = (b.cx - padX) / scale, cy = (b.cy - padY) / scale;
      const w = b.w / scale, h = b.h / scale;
      return {
        cls: b.cls,
        score: b.score,
        cx, cy,
        x1: Math.max(0, cx - w / 2),
        y1: Math.max(0, cy - h / 2),
        x2: Math.min(img.width, cx + w / 2),
        y2: Math.min(img.height, cy + h / 2)
      };
    });
  } else {
    setStatus("Neural model is still loading weights. Please wait a moment…", "warn");
    spinner.classList.add("hidden");
    btnText.textContent = "⚡ Run Follicular Analysis";
    return;
  }

  currentDets = dets;
  const r = (lastResult = summarize(dets, img, pxmm));
  const out = $("out");
  draw(out, img, dets);

  // Update KPI Stat Cards
  $("statTotal").textContent = r.total;
  $("statTotalSub").textContent = `${r.counts.fu1} fu1 · ${r.counts.fu2} fu2 · ${r.counts.fu3} fu3 · ${r.counts.fu4} fu4 · ${r.counts.fu5} fu5`;
  $("statMean").textContent = r.mean.toFixed(2);
  $("statDensity").textContent = r.density ? r.density.toFixed(1) : "—";
  $("statDensitySub").textContent = r.areaCm2
    ? `Visible field: ${r.areaCm2.toFixed(1)} cm²`
    : "Enter px/mm scale above to enable physical density";

  const norwoodBadge = $("statNorwoodBadge");
  norwoodBadge.textContent = r.norwoodTitle;
  $("statNorwoodSub").textContent = r.norwoodSub;

  // Render Zone Bars
  const maxZoneCount = Math.max(...r.bands.map(b => b.count), 1);
  $("zoneBars").innerHTML = r.bands.map(b => {
    const pct = Math.round((b.count / maxZoneCount) * 100);
    const fu1Pct = (b.fu1Share * 100).toFixed(0);
    return `
      <div class="zone-bar-item">
        <div class="zone-bar-header">
          <span class="zone-bar-name">${b.name[0].toUpperCase() + b.name.slice(1)} Zone</span>
          <span class="zone-bar-count">${b.count} units ${b.density ? `(${b.density.toFixed(1)} FU/cm²)` : ""}</span>
        </div>
        <div class="zone-bar-track">
          <div class="zone-bar-fill" style="width: ${pct}%"></div>
        </div>
        <div class="zone-bar-sub">Single-hair miniaturization: ${fu1Pct}% fu1</div>
      </div>
    `;
  }).join("");

  // Render Graft Table
  $("planBody").innerHTML = r.rows.map(row => `
    <tr>
      <td>${row.zone}</td>
      <td>${row.area.toFixed(0)}</td>
      <td>${row.zd ? row.zd.toFixed(1) : "—"}</td>
      <td>${row.deficit.toFixed(1)}</td>
      <td>${row.target}</td>
      <td style="color: #10b981; font-weight: 600;">${row.allocated}</td>
    </tr>
  `).join("");

  drawChart($("chart"), r.rows);

  $("assumptions").textContent = "Clinical Assumptions: " + r.supplyNote +
    ` · target ${TARGET} FU/cm² · safe donor harvest ${HARVEST * 100}% · ` +
    (r.areaCm2 ? "zone areas calculated from image calibration" : "zone areas assumed 45/45/40 cm² (uncalibrated)") +
    "\nGreedy deterministic allocation — designed as a research sandbox, not a surgical prescription.";

  $("results").classList.remove("hidden");
  spinner.classList.add("hidden");
  btnText.textContent = "⚡ Run Follicular Analysis";
}

/* ---------- Photo Display & Thumbnail Card ---------- */
function showImagePreview(imageUrl, title, dimensions) {
  $("thumbImg").src = imageUrl;
  $("previewTitle").textContent = title;
  $("previewDimensions").textContent = dimensions;

  $("drop").classList.add("hidden");
  $("photoPreviewCard").classList.remove("hidden");
}

function resetPhoto() {
  img = null;
  cachedInference = null;
  lastResult = null;
  $("photoPreviewCard").classList.add("hidden");
  $("drop").classList.remove("hidden");
  $("results").classList.add("hidden");
  setStatus(session ? "Ready: Select or drop a scalp photo to analyze." : "Initializing model…", session ? "ready" : "pulse");
}

/* ---------- File Handling ---------- */
async function handleFile(file) {
  cachedInference = null;
  lastResult = null;
  currentPhotoName = file.name || "scalp-photo.jpg";
  const url = URL.createObjectURL(file);
  img = await loadImageFromSource(url);

  showImagePreview(url, currentPhotoName, `${img.width} × ${img.height} px`);
  $("results").classList.add("hidden");

  if (session) {
    analyze(true);
  } else {
    setStatus("Photo loaded in memory. Downloading neural weights… will analyze automatically when ready.", "pulse");
  }
}

/* ---------- Export Utilities ---------- */
function downloadAnnotatedCanvas() {
  const canvas = $("out");
  const link = document.createElement("a");
  link.download = `openhairstack-${currentPhotoName.replace(/\.[^/.]+$/, "")}-annotated.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function exportJsonReport() {
  if (!lastResult) return;
  const report = {
    project: "Open Hair Stack",
    generator: "Scalp Analyzer v0.1 (follicle-fu-v0.1 ONNX)",
    timestamp: new Date().toISOString(),
    photo: {
      name: currentPhotoName,
      width: img ? img.width : null,
      height: img ? img.height : null,
      calibration_px_per_mm: parseFloat($("pxmm").value) || null
    },
    results: {
      total_follicular_units: lastResult.total,
      mean_hairs_per_unit: lastResult.mean,
      measured_density_fu_cm2: lastResult.density,
      class_counts: lastResult.counts,
      heuristic_norwood: {
        pattern: lastResult.norwoodTitle,
        details: lastResult.norwoodSub
      },
      zone_distribution: lastResult.bands,
      graft_simulation: lastResult.rows
    },
    disclaimer: "Research demonstration only. Not a medical device or diagnostic treatment plan."
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = `openhairstack-report-${Date.now()}.json`;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

/* ---------- UI Wiring ---------- */
function wire() {
  // File Picker
  $("pickBtn").onclick = () => $("fileInput").click();
  $("repickBtn").onclick = () => $("fileInput").click();
  $("clearPhotoBtn").onclick = resetPhoto;

  $("fileInput").onchange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  // Drag & Drop
  const drop = $("drop");
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("over");
  }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  // Benchmark Sample Buttons
  document.querySelectorAll("[data-sample]").forEach(b => b.onclick = async () => {
    cachedInference = null;
    lastResult = null;
    currentPhotoName = b.dataset.name || "Benchmark Sample";
    const src = b.dataset.sample;
    img = await loadImageFromSource(src);

    showImagePreview(src, currentPhotoName, `${img.width} × ${img.height} px`);
    if (session) {
      analyze(true);
    } else {
      setStatus("Benchmark sample loaded. Neural weights downloading in background…", "pulse");
    }
  });

  // Sliders & Calibration
  $("conf").oninput = (e) => {
    $("confVal").textContent = e.target.value;
    if (img && cachedInference) analyze(false);
  };

  $("pxmm").oninput = () => {
    const val = parseFloat($("pxmm").value);
    $("calibStatus").textContent = (val && val >= 1) ? `${val.toFixed(1)} px/mm Calibrated` : "Uncalibrated";
    if (img && cachedInference) analyze(false);
  };

  document.querySelectorAll(".preset-btn").forEach(btn => btn.onclick = () => {
    const val = btn.dataset.pxmm;
    $("pxmm").value = val;
    $("calibStatus").textContent = val ? `${val} px/mm Calibrated` : "Uncalibrated";
    if (img && cachedInference) analyze(false);
  });

  // Filter Chips
  document.querySelectorAll(".chip").forEach(chip => chip.onclick = () => {
    document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    activeClassFilter = chip.dataset.filter;
    if (img && currentDets.length) {
      draw($("out"), img, currentDets);
    }
  });

  // Action Buttons
  $("analyze").onclick = () => analyze(true);
  $("quickAnalyzeBtn").onclick = () => analyze(true);
  $("downloadImageBtn").onclick = downloadAnnotatedCanvas;
  $("exportJsonBtn").onclick = exportJsonReport;
}

wire();
loadModel();
