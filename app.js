/* Open Hair Stack — client-side follicular-unit analyzer.
   Model runs in the browser via onnxruntime-web. No uploads, no server.

   Pipeline: letterbox to 640x640 -> ONNX session -> transpose output ->
   class scores (sigmoid-activated in exported graph) -> NMS -> map back to
   image coords -> draw, count, band-split, heuristic, plan simulation.
*/
const MODEL_URL = "https://huggingface.co/openhairstack/follicle-fu-v0.1/resolve/main/model.onnx";
const CLASSES = ["fu1", "fu2", "fu3", "fu4", "fu5"];
const HAIRS = { fu1: 1, fu2: 2, fu3: 3, fu4: 4, fu5: 5 };  // hair count per class in v0.1
const COLORS = { fu1: "#e6194b", fu2: "#f78230", fu3: "#3cb44b", fu4: "#0082c8", fu5: "#808080" };
const TARGET = 25, DONOR_AREA = 150, HARVEST = 0.3, ZONE_AREAS = [45, 45, 40];
const ZONE_NAMES = ["anterior", "middle", "posterior"];

// Configure ONNX Web runtime for broad browser compatibility (single-threaded WASM with SIMD)
if (typeof ort !== "undefined" && ort.env && ort.env.wasm) {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
}

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let session = null, img = null, lastResult = null, cachedInference = null;

const setStatus = (msg, cls) => {
  statusEl.textContent = msg;
  statusEl.className = "status" + (cls ? " " + cls : "");
};

/* ---------- model ---------- */
async function loadModel() {
  setStatus("Model: downloading follicle-fu-v0.1 (~10 MB, runs client-side via WASM)…");
  try {
    const resp = await fetch(MODEL_URL, { method: "HEAD" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  } catch (e) {
    setStatus("Model weights not reachable (" + e.message + "). The UI operates in preview mode.", "err");
    return;
  }
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    $("analyze").disabled = false;
    setStatus("Model: follicle-fu-v0.1 loaded — ready. Runs 100% locally in your browser via WASM.", "ok");
    // If a photo was selected before the model finished loading, run inference now
    if (img && !lastResult) {
      analyze(true);
    }
  } catch (e) {
    console.error("ONNX initialization error:", e);
    setStatus("Model failed to initialize: " + e.message + ". You can still explore the interface with sample images.", "err");
  }
}

/* ---------- image loading ---------- */
function loadImageFromSource(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error("Could not load image"));
    im.src = src;
  });
}

/* ---------- preprocess + inference ---------- */
function letterbox(image) {
  const S = 640, canvas = document.createElement("canvas");
  canvas.width = canvas.height = S;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#727272";
  ctx.fillRect(0, 0, S, S);
  const scale = Math.min(S / image.width, S / image.height);
  const w = image.width * scale, h = image.height * scale;
  ctx.drawImage(image, (S - w) / 2, (S - h) / 2, w, h);
  const data = ctx.getImageData(0, 0, S, S).data;
  const planes = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    planes[i] = data[i * 4] / 255;                     // R
    planes[S * S + i] = data[i * 4 + 1] / 255;         // G
    planes[2 * S * S + i] = data[i * 4 + 2] / 255;     // B
  }
  return { planes, scale, padX: (S - w) / 2, padY: (S - h) / 2 };
}

function transposeAndDecode(output) {
  // output: (1, 4+nc, N) rows OR (1, N, 4+nc) cols
  const dims = output.dims;
  const nc = CLASSES.length;
  let boxes = [];
  if (dims[1] === 4 + nc) {           // rows layout: [1, 9, 8400]
    const N = dims[2], d = output.data;
    for (let j = 0; j < N; j++) {
      const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
      let best = -1, bestCls = 0;
      for (let c = 0; c < nc; c++) {
        const s = d[(4 + c) * N + j];  // scores are sigmoid-activated in the ONNX graph
        if (s > best) { best = s; bestCls = c; }
      }
      if (best >= 0.005) boxes.push({ cx, cy, w, h, score: best, cls: bestCls });
    }
  } else {                            // columns layout: [1, 8400, 9]
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
    // Class-agnostic suppression to avoid duplicate detections on the same pore
    boxes = boxes.filter((o) => !(iou(o, b) > iouThresh));
  }
  return keep;
}

/* ---------- stats + heuristic + plan ---------- */
function summarize(dets, image, pxmm) {
  const counts = {}; CLASSES.forEach(c => counts[c] = 0);
  dets.forEach(d => counts[CLASSES[d.cls]]++);
  const total = dets.length;
  const mean = total ? dets.reduce((s, d) => s + HAIRS[CLASSES[d.cls]], 0) / total : 0;
  const known = pxmm && pxmm > 3;
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

  let norwood = "inconclusive (no units detected — try adjusting confidence slider or use a clear macro crop)";
  if (total > 0) {
    if (bands[2].count > 0 && bands[0].count === 0) {
      norwood = "marked anterior deficit with preserved posterior density (illustrative: NW ~3–4 pattern)";
    } else if (bands[2].count && bands[0].count) {
      const ratio = (bands[0].fu1Share + 0.05) / (bands[2].fu1Share + 0.05);
      if (ratio > 1.6 || bands[0].count < bands[2].count * 0.6)
        norwood = "consistent with significant frontal thinning (illustrative: NW ~3–4)";
      else if (ratio > 1.25 || bands[0].count < bands[2].count * 0.85)
        norwood = "possible frontal thinning (illustrative: NW ~2)";
      else norwood = "no obvious anterior/posterior gradient in this crop (illustrative: NW 1)";
    } else {
      norwood = "follicular units localized in visible crop; uniform distribution across detected zones";
    }
  }

  const capacity = density ? Math.round(density * DONOR_AREA * HARVEST) : 1125;
  const supplyNote = density
    ? `donor capacity est. ${capacity} FU (measured density ${density.toFixed(1)}/cm²)`
    : "donor capacity assumed ~25 FU/cm² × 150 cm² × 30% harvest";
  const rows = bands.map((b, i) => {
    const area = b.area || ZONE_AREAS[i];
    const zd = b.density || 20;
    const deficit = Math.max(0, TARGET - zd);
    return { zone: b.name[0].toUpperCase() + b.name.slice(1), area, zd: b.density,
             deficit, target: Math.round(area * deficit), allocated: 0 };
  });
  const demand = rows.reduce((s, r) => s + r.target, 0);
  const scale = demand > 0 ? Math.min(1, capacity / demand) : 1;
  rows.forEach(r => r.allocated = Math.round(r.target * scale));

  return { counts, total, mean, areaCm2, density, bands, norwood, rows, capacity, supplyNote, scale };
}

function draw(canvas, image, dets) {
  const ctx = canvas.getContext("2d");
  canvas.width = image.width; canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
  dets.forEach(d => {
    const cls = CLASSES[d.cls];
    ctx.strokeStyle = COLORS[cls]; ctx.lineWidth = 2.5;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    ctx.fillStyle = COLORS[cls]; ctx.font = "bold 13px 'Segoe UI', sans-serif";
    ctx.fillText(`${cls} ${d.score.toFixed(2)}`, d.x1, Math.max(14, d.y1 - 4));
  });

  // Dark translucent HUD box for count overlay
  const hudW = 120, hudH = CLASSES.length * 20 + 12;
  ctx.fillStyle = "rgba(10, 16, 26, 0.82)";
  ctx.strokeStyle = "rgba(42, 58, 82, 0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(8, 8, hudW, hudH, 6);
  else ctx.rect(8, 8, hudW, hudH);
  ctx.fill();
  ctx.stroke();

  let y = 26;
  ctx.font = "13px 'Segoe UI', sans-serif";
  CLASSES.forEach(c => {
    ctx.fillStyle = COLORS[c]; ctx.fillRect(16, y - 10, 10, 10);
    ctx.fillStyle = "#e8ecf4"; ctx.fillText(`${c}: ${lastResult.counts[c]}`, 34, y);
    y += 20;
  });
}

function drawChart(cv, rows) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...rows.map(r => r.target), 1);
  rows.forEach((r, i) => {
    const y = 20 + i * 50, barW = 230;
    ctx.fillStyle = "#9fb0cc"; ctx.font = "13px 'Segoe UI', sans-serif";
    ctx.fillText(r.zone, 6, y + 14);
    // Track backgrounds
    ctx.strokeStyle = "#2a3a52";
    ctx.strokeRect(110, y, barW, 11);
    ctx.strokeRect(110, y + 15, barW, 11);
    // Green = allocated
    ctx.fillStyle = "#3cb44b";
    ctx.fillRect(110, y, barW * (r.allocated / max), 11);
    // Blue = target
    ctx.fillStyle = "#4da3ff";
    ctx.fillRect(110, y + 15, barW * (r.target / max), 11);
    // Label numbers
    ctx.fillStyle = "#e8ecf4";
    ctx.fillText(`${r.allocated} / ${r.target}`, 110 + barW + 10, y + 14);
  });
  ctx.fillStyle = "#9fb0cc"; ctx.font = "11px 'Segoe UI', sans-serif";
  ctx.fillText("green = allocated · blue = zone target", 110, H - 6);
}

/* ---------- analysis flow ---------- */
async function runInferenceOnImage(image) {
  const { planes, scale, padX, padY } = letterbox(image);
  const input = new ort.Tensor("float32", planes, [1, 3, 640, 640]);
  const feeds = {}; feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  const rawBoxes = transposeAndDecode(out[session.outputNames[0]]);
  return { rawBoxes, scale, padX, padY };
}

async function analyze(forceNewInference = true) {
  if (!img) return;
  const conf = parseFloat($("conf").value);
  const pxmm = parseFloat($("pxmm").value) || null;
  let dets = [];

  if (session) {
    if (forceNewInference || !cachedInference) {
      setStatus("Running neural inference in browser…", "ok");
      cachedInference = await runInferenceOnImage(img);
      setStatus("Model: follicle-fu-v0.1 loaded — ready. Runs 100% locally in your browser via WASM.", "ok");
    }
    const { rawBoxes, scale, padX, padY } = cachedInference;
    const candidates = rawBoxes.filter(b => b.score >= conf);
    const kept = nms(candidates, 0.45).slice(0, 400);
    dets = kept.map(b => {
      const cx = (b.cx - padX) / scale, cy = (b.cy - padY) / scale;
      const w = b.w / scale, h = b.h / scale;
      return {
        cls: b.cls, score: b.score, cx, cy,
        x1: Math.max(0, cx - w / 2),
        y1: Math.max(0, cy - h / 2),
        x2: Math.min(img.width, cx + w / 2),
        y2: Math.min(img.height, cy + h / 2)
      };
    });
  } else {
    setStatus("Model is still downloading weights. Please wait a moment, then click Analyze.", "warn");
    return;
  }

  const r = (lastResult = summarize(dets, img, pxmm));
  const out = $("out");
  draw(out, img, dets);

  let html = `<b>${r.total}</b> follicular units detected · mean hairs/unit <b>${r.mean.toFixed(2)}</b>\n\n`;
  html += CLASSES.map(c => `${c}: ${r.counts[c]}`).join("  ·  ") + "\n";
  if (r.areaCm2) html += `\nVisible area ${r.areaCm2.toFixed(1)} cm² · density <b>${r.density.toFixed(1)} FU/cm²</b>`;
  html += "\n\n" + r.bands.map(b =>
    `${b.name}: ${b.count} units${b.density ? `, ${b.density.toFixed(1)}/cm²` : ""}, ${(b.fu1Share * 100).toFixed(0)}% single-hair`
  ).join("\n");
  html += `\n\n<b>Heuristic readout:</b> ${r.norwood}`;
  $("summary").innerHTML = html.replace(/\n/g, "<br/>");

  $("planBody").innerHTML = r.rows.map(row =>
    `<tr><td>${row.zone}</td><td>${row.area.toFixed(0)}</td><td>${row.zd ? row.zd.toFixed(1) : "—"}</td>` +
    `<td>${row.deficit.toFixed(1)}</td><td>${row.target}</td><td>${row.allocated}</td></tr>`).join("");
  drawChart($("chart"), r.rows);

  $("assumptions").textContent = "Assumptions: " + r.supplyNote +
    ` · target ${TARGET} FU/cm² · harvest ${HARVEST * 100}% of donor · ` +
    (r.areaCm2 ? "zone areas measured" : "zone areas assumed 45/45/40 cm² (no calibration)") +
    "\nAllocation is greedy and transparent by design — it is an educational sandbox, not a surgical plan.";
  $("results").classList.remove("hidden");
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- UI wiring ---------- */
function wire() {
  $("pickBtn").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { if (e.target.files[0]) onFile(e.target.files[0]); };
  const drop = $("drop");
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) onFile(f); });

  $("conf").oninput = (e) => {
    $("confVal").textContent = e.target.value;
    if (img && cachedInference) analyze(false);
  };
  $("pxmm").oninput = () => {
    if (img && cachedInference) analyze(false);
  };

  document.querySelectorAll("[data-sample]").forEach(b => b.onclick = async () => {
    cachedInference = null;
    lastResult = null;
    img = await loadImageFromSource(b.dataset.sample);
    if (session) {
      analyze(true);
    } else {
      setStatus("Sample loaded. Neural model is downloading in background… will analyze automatically when ready.", "ok");
    }
  });
  $("analyze").onclick = () => analyze(true);
}

async function onFile(file) {
  cachedInference = null;
  lastResult = null;
  const url = URL.createObjectURL(file);
  img = await loadImageFromSource(url);
  $("results").classList.add("hidden");
  if (session) {
    setStatus("Image loaded. Click 'Analyze' to run detection.", "ok");
  } else {
    setStatus("Image loaded. Neural model is still loading weights in background…", "ok");
  }
}

wire();
loadModel();
