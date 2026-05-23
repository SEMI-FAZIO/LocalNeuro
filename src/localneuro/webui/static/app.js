/* ===================================================================
   LocalNeuro web UI -- frontend logic (vanilla JS, no framework)
   =================================================================== */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const appState = { model: null, device: "" };
let chatHistory = [];           // [{role, content}]
let pgController = null;        // AbortController for playground generation
let trainController = null;     // AbortController for the training event stream
let chartData = { train: [], val: [] };

/* --------------------------------------------------------------- utils */
function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value == null ? "" : String(value);
  return node.innerHTML;
}

function fmtBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return n.toFixed(i ? 1 : 0) + " " + units[i];
}

function fmtNum(value) {
  if (value === undefined || value === null) return "-";
  return typeof value === "number" ? value.toFixed(4) : String(value);
}

let toastTimer = null;
function toast(message, isError) {
  const el = $("#toast");
  el.textContent = message;
  el.className = "toast show" + (isError ? " toast--error" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "toast"; }, 3800);
}

/* ----------------------------------------------------------- networking */
async function api(method, path, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const resp = await fetch(path, init);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || ("HTTP " + resp.status));
  return data;
}

/* Stream a chunked NDJSON response, invoking onEvent for each JSON line. */
async function streamRequest(path, opts, onEvent) {
  opts = opts || {};
  const init = { method: opts.method || "POST", signal: opts.signal };
  if (opts.body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, init);
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.error || ("HTTP " + resp.status));
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      try { onEvent(JSON.parse(line)); } catch (err) { /* skip partial */ }
    }
  }
  const tail = (buffer + decoder.decode()).trim();
  if (tail) { try { onEvent(JSON.parse(tail)); } catch (err) { /* ignore */ } }
}

/* ------------------------------------------------------------- views */
function switchView(name) {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "models") loadCheckpoints();
  if (name === "training") refreshTraining();
}

/* --------------------------------------------------------- sliders */
function bindSlider(id) {
  const input = $("#" + id);
  const label = $("#" + id + "-val");
  if (!input || !label) return;
  const decimals = String(input.step).includes(".");
  const render = () => {
    label.textContent = decimals
      ? parseFloat(input.value).toFixed(2)
      : input.value;
  };
  input.addEventListener("input", render);
  render();
}

function readSampling(prefix) {
  return {
    temperature: parseFloat($("#" + prefix + "-temp").value),
    top_k: parseInt($("#" + prefix + "-topk").value, 10),
    top_p: parseFloat($("#" + prefix + "-topp").value),
    repetition_penalty: parseFloat($("#" + prefix + "-reppen").value),
  };
}

/* ----------------------------------------------------- status / header */
async function refreshStatus() {
  try {
    const status = await api("GET", "/api/status");
    appState.model = status.model;
    appState.device = status.device;
  } catch (err) { /* server not ready yet */ }
  renderActiveModel();
}

function renderActiveModel() {
  const chip = $("#active-model");
  const device = $("#device-info");
  if (appState.model) {
    chip.classList.remove("model-chip--empty");
    chip.innerHTML = "<b>" + escapeHtml(appState.model.name) + "</b>";
    device.textContent =
      appState.model.params_human + " params · " + appState.device;
  } else {
    chip.classList.add("model-chip--empty");
    chip.textContent = "No model loaded";
    device.textContent = appState.device ? "device: " + appState.device : "";
  }
}

/* ------------------------------------------------------------- chat */
function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-" + role;
  const roleEl = document.createElement("div");
  roleEl.className = "msg-role";
  roleEl.textContent = role === "user" ? "You" : "LocalNeuro";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  wrap.appendChild(roleEl);
  wrap.appendChild(body);
  $("#chat-messages").appendChild(wrap);
  scrollChat();
  return body;
}

function scrollChat() {
  const box = $("#chat-messages");
  box.scrollTop = box.scrollHeight;
}

function clearChat() {
  chatHistory = [];
  $("#chat-messages").innerHTML =
    '<div class="placeholder" id="chat-placeholder">Load a model from the ' +
    "<b>Models</b> tab, then start chatting.</div>";
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 180) + "px";
}

async function sendChat() {
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  if (!appState.model) { toast("Load a model first", true); return; }

  input.value = "";
  autoGrow(input);
  const placeholder = $("#chat-placeholder");
  if (placeholder) placeholder.remove();

  addMessage("user", text);
  const body = addMessage("assistant", "");
  body.classList.add("typing");
  $("#chat-send").disabled = true;

  let reply = "";
  try {
    await streamRequest("/api/chat", {
      body: {
        message: text,
        history: chatHistory.slice(),
        system: $("#chat-system").value.trim(),
        sampling: readSampling("chat"),
        max_new_tokens: parseInt($("#chat-maxtok").value, 10),
      },
    }, (event) => {
      if (event.type === "token") {
        reply += event.text;
        body.textContent = reply;
        scrollChat();
      } else if (event.type === "error") {
        toast(event.message, true);
      }
    });
  } catch (err) {
    toast(err.message, true);
  } finally {
    body.classList.remove("typing");
    $("#chat-send").disabled = false;
  }

  if (!reply) body.textContent = "(no output)";
  chatHistory.push({ role: "user", content: text });
  chatHistory.push({ role: "assistant", content: reply });
}

/* -------------------------------------------------------- playground */
async function runGenerate() {
  if (!appState.model) { toast("Load a model first", true); return; }
  const prompt = $("#pg-prompt").value;
  const output = $("#pg-output");
  output.innerHTML = escapeHtml(prompt) + '<span class="gen"></span>';
  const genSpan = output.querySelector(".gen");

  const seedRaw = $("#pg-seed").value.trim();
  pgController = new AbortController();
  $("#pg-generate").disabled = true;
  $("#pg-stop").disabled = false;

  let generated = "";
  try {
    await streamRequest("/api/generate", {
      body: {
        prompt: prompt,
        sampling: readSampling("pg"),
        max_new_tokens: parseInt($("#pg-maxtok").value, 10),
        seed: seedRaw === "" ? null : parseInt(seedRaw, 10),
      },
      signal: pgController.signal,
    }, (event) => {
      if (event.type === "token") {
        generated += event.text;
        genSpan.textContent = generated;
        output.scrollTop = output.scrollHeight;
      } else if (event.type === "error") {
        toast(event.message, true);
      }
    });
  } catch (err) {
    if (err.name !== "AbortError") toast(err.message, true);
  } finally {
    $("#pg-generate").disabled = false;
    $("#pg-stop").disabled = true;
    pgController = null;
  }
}

function stopGenerate() {
  if (pgController) pgController.abort();
}

/* ------------------------------------------------------------ models */
async function loadCheckpoints() {
  try {
    const data = await api("GET", "/api/checkpoints");
    renderCheckpoints(data.checkpoints);
    populateQuantSource(data.checkpoints);
    populateFinetuneBase(data.checkpoints);
    populateTrainPresets(data.configs);
  } catch (err) {
    $("#models-list").innerHTML =
      '<div class="placeholder">Could not load checkpoints.</div>';
  }
}

function renderCheckpoints(checkpoints) {
  const list = $("#models-list");
  if (!checkpoints.length) {
    list.innerHTML =
      '<div class="placeholder">No checkpoints yet. Train one from the ' +
      "<b>Training</b> tab.</div>";
    return;
  }
  list.innerHTML = "";
  checkpoints.forEach((ckpt) => {
    const card = document.createElement("div");
    card.className = "ckpt" + (ckpt.active ? " active" : "");

    const quant = ckpt.quantization
      ? '<span class="badge badge--quant">int' + ckpt.quantization + "</span>"
      : "";
    const sft = ckpt.stage === "sft"
      ? '<span class="badge badge--sft">SFT</span>' : "";
    const active = ckpt.active
      ? '<span class="badge badge--active">active</span>' : "";
    const step = ckpt.step != null ? "step " + ckpt.step + " · " : "";

    const info = document.createElement("div");
    info.className = "ckpt-info";
    info.innerHTML =
      '<div class="ckpt-name">' + escapeHtml(ckpt.name) + quant + sft + active + "</div>" +
      '<div class="ckpt-meta">' + ckpt.params_human + " params · " +
      ckpt.n_layers + "L · ctx " + ckpt.max_seq_len + " · " +
      step + fmtBytes(ckpt.size_bytes) + "</div>";

    const button = document.createElement("button");
    button.className = "btn " + (ckpt.active ? "btn-ghost" : "btn-primary");
    button.textContent = ckpt.active ? "Loaded" : "Load";
    button.disabled = ckpt.active;
    button.addEventListener("click", () => loadModel(ckpt.path));

    card.appendChild(info);
    card.appendChild(button);
    list.appendChild(card);
  });
}

async function loadModel(path) {
  toast("Loading model...");
  try {
    const result = await api("POST", "/api/load", { path });
    appState.model = result.model;
    renderActiveModel();
    loadCheckpoints();
    toast("Model loaded");
  } catch (err) {
    toast(err.message, true);
  }
}

function populateQuantSource(checkpoints) {
  const select = $("#quant-source");
  const current = select.value;
  select.innerHTML = "";
  checkpoints.forEach((ckpt) => {
    const option = document.createElement("option");
    option.value = ckpt.path;
    option.textContent = ckpt.name;
    select.appendChild(option);
  });
  if (current) select.value = current;
}

async function runQuantize() {
  const source = $("#quant-source").value;
  const name = $("#quant-output").value.trim();
  const bits = parseInt($("#quant-bits").value, 10);
  const result = $("#quant-result");
  if (!source) { toast("Pick a source checkpoint", true); return; }
  if (!name) { toast("Enter an output name", true); return; }

  result.innerHTML = '<span class="muted">Quantizing...</span>';
  try {
    const data = await api("POST", "/api/quantize", {
      input: source,
      output: "checkpoints/" + name,
      bits: bits,
    });
    result.innerHTML =
      '<span class="ok">Done.</span> ' + data.layers + " layers · " +
      fmtBytes(data.size_before) + " → " + fmtBytes(data.size_after) +
      " → <b>" + escapeHtml(data.output) + "</b>";
    loadCheckpoints();
  } catch (err) {
    result.innerHTML = '<span class="err">' + escapeHtml(err.message) + "</span>";
  }
}

/* ---------------------------------------------------------- training */
function populateTrainPresets(configs) {
  const select = $("#train-preset");
  const current = select.value;
  select.innerHTML = "";
  (configs || []).forEach((cfg) => {
    const option = document.createElement("option");
    option.value = cfg.name;
    option.textContent =
      cfg.name + " (" + cfg.params_human + ", " + cfg.n_layers +
      "L, ctx " + cfg.max_seq_len + ")";
    select.appendChild(option);
  });
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = "custom architecture...";
  select.appendChild(custom);
  if (current) select.value = current;
}

function onPresetChange() {
  const isCustom = $("#train-preset").value === "custom";
  $("#train-custom").classList.toggle("hidden", !isCustom);
}

function populateFinetuneBase(checkpoints) {
  const select = $("#train-base");
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  // A quantized checkpoint cannot be fine-tuned -- its weights are frozen.
  const usable = (checkpoints || []).filter((c) => !c.quantization);
  if (!usable.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "(no base checkpoints -- pretrain one first)";
    select.appendChild(option);
    return;
  }
  usable.forEach((ckpt) => {
    const option = document.createElement("option");
    option.value = ckpt.path;
    option.textContent = ckpt.name + "  (" + ckpt.params_human + ")";
    select.appendChild(option);
  });
  if (current) select.value = current;
}

function onTrainModeChange() {
  const finetune = $("#train-mode").value === "finetune";
  $("#pretrain-fields").classList.toggle("hidden", finetune);
  $("#finetune-fields").classList.toggle("hidden", !finetune);
  // The two modes need very different learning rates and block sizes.
  $("#train-lr").value = finetune ? "0.00005" : "0.0006";
  $("#train-block").value = finetune ? "256" : "128";
}

function setTrainStatus(status) {
  const badge = $("#train-status");
  badge.textContent = status;
  badge.className = "status-badge status--" + status;
  const running = status === "running" || status === "preparing";
  $("#train-start").disabled = running;
  $("#train-stop").disabled = !running;
}

function logLine(text, cls) {
  const log = $("#train-log");
  const atBottom =
    log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  const line = document.createElement("div");
  line.className = "log-" + (cls || "msg");
  line.textContent = text;
  log.appendChild(line);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function handleTrainLog(record) {
  if (record.message !== undefined) {
    logLine(record.message, "msg");
    return;
  }
  if (record.loss !== undefined && record.step !== undefined) {
    chartData.train.push({ x: record.step, y: record.loss });
    logLine(
      "step " + record.step + "   loss " + fmtNum(record.loss) +
      "   ppl " + fmtNum(record.ppl) + "   lr " + fmtNum(record.lr),
      "metric");
    drawChart();
  }
  if (record.val_loss !== undefined && record.step !== undefined) {
    chartData.val.push({ x: record.step, y: record.val_loss });
    logLine(
      "step " + record.step + "   val_loss " + fmtNum(record.val_loss) +
      "   val_ppl " + fmtNum(record.val_ppl),
      "metric");
    drawChart();
  }
}

function handleTrainEvent(event) {
  if (event.type === "ping") return;
  if (event.type === "status") {
    setTrainStatus(event.status);
  } else if (event.type === "log") {
    handleTrainLog(event.record || {});
  } else if (event.type === "done") {
    setTrainStatus(event.status);
    if (event.status === "error") {
      logLine("error: " + (event.error || "unknown"), "err");
      toast("Training failed: " + (event.error || "unknown"), true);
    } else {
      logLine("=== training " + event.status + " ===", "msg");
      toast("Training " + event.status);
    }
    trainController = null;
    loadCheckpoints();
  }
}

function streamTraining() {
  if (trainController) trainController.abort();
  trainController = new AbortController();
  streamRequest("/api/train/events",
    { method: "GET", signal: trainController.signal },
    handleTrainEvent
  ).catch((err) => { /* aborted or disconnected */ });
}

async function refreshTraining() {
  try {
    const snap = await api("GET", "/api/train/status");
    resetChart();
    $("#train-log").innerHTML = "";
    if (snap.status === "running" || snap.status === "preparing") {
      // subscribe() replays the full history, then streams live events.
      streamTraining();
    } else {
      (snap.history || []).forEach(handleTrainEvent);
      setTrainStatus(snap.status);
      if (snap.status === "error" && snap.error) {
        logLine("error: " + snap.error, "err");
      }
    }
  } catch (err) { /* ignore */ }
}

async function startTraining() {
  const mode = $("#train-mode").value;
  const shared = {
    run_name: $("#train-runname").value.trim() || "gui-run",
    steps: parseInt($("#train-steps").value, 10),
    batch_size: parseInt($("#train-batch").value, 10),
    block_size: parseInt($("#train-block").value, 10),
    learning_rate: parseFloat($("#train-lr").value),
  };
  let params;
  if (mode === "finetune") {
    const base = $("#train-base").value;
    if (!base) {
      toast("No base checkpoint -- pretrain one first", true);
      return;
    }
    params = Object.assign({}, shared, {
      mode: "finetune",
      base_checkpoint: base,
      synthetic_instructions: parseInt($("#train-instructions").value, 10),
    });
  } else {
    const preset = $("#train-preset").value;
    params = Object.assign({}, shared, {
      mode: "pretrain",
      preset: preset,
      vocab_size: parseInt($("#train-vocab").value, 10),
      synthetic_samples: parseInt($("#train-synthetic").value, 10),
      use_demo_corpus: $("#train-demo").checked,
    });
    if (preset === "custom") {
      params.d_model = parseInt($("#train-dmodel").value, 10);
      params.n_layers = parseInt($("#train-layers").value, 10);
      params.n_heads = parseInt($("#train-heads").value, 10);
      params.max_seq_len = parseInt($("#train-maxseq").value, 10);
    }
  }
  try {
    await api("POST", "/api/train/start", params);
    resetChart();
    $("#train-log").innerHTML = "";
    setTrainStatus("preparing");
    toast("Training started");
    streamTraining();
  } catch (err) {
    toast(err.message, true);
  }
}

async function stopTraining() {
  try {
    await api("POST", "/api/train/stop");
    toast("Stopping at the next step...");
  } catch (err) {
    toast(err.message, true);
  }
}

/* ------------------------------------------------------- loss chart */
function resetChart() {
  chartData = { train: [], val: [] };
  drawChart();
}

function drawChart() {
  const canvas = $("#loss-chart");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;  // view hidden

  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { l: 48, r: 14, t: 14, b: 26 };
  ctx.clearRect(0, 0, W, H);

  const points = chartData.train.concat(chartData.val);
  if (!points.length) {
    ctx.fillStyle = "#686a76";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("waiting for training metrics...", W / 2, H / 2);
    ctx.textAlign = "left";
    return;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMax = Math.max(1, Math.max.apply(null, xs));
  let yMin = Math.min.apply(null, ys);
  let yMax = Math.max.apply(null, ys);
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
  const margin = (yMax - yMin) * 0.12;
  yMin = Math.max(0, yMin - margin);
  yMax = yMax + margin;

  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const sx = (x) => pad.l + (x / xMax) * plotW;
  const sy = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  // grid + y-axis labels
  ctx.strokeStyle = "#23252f";
  ctx.fillStyle = "#686a76";
  ctx.font = "11px monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
    ctx.fillText((yMax - (i / 4) * (yMax - yMin)).toFixed(2), 6, y + 3);
  }
  ctx.fillText("0", pad.l, H - 8);
  ctx.fillText("step " + Math.round(xMax), W - pad.r - 60, H - 8);

  const drawSeries = (series, color) => {
    if (!series.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((p, i) => {
      const x = sx(p.x), y = sy(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawSeries(chartData.train, "#5ad19b");
  drawSeries(chartData.val, "#e0b66b");
}

/* -------------------------------------------------------------- init */
function init() {
  $$(".nav-item").forEach((btn) =>
    btn.addEventListener("click", () => switchView(btn.dataset.view)));

  ["chat-temp", "chat-topk", "chat-topp", "chat-reppen", "chat-maxtok",
   "pg-temp", "pg-topk", "pg-topp", "pg-reppen", "pg-maxtok"].forEach(bindSlider);

  // chat
  $("#chat-send").addEventListener("click", sendChat);
  $("#chat-clear").addEventListener("click", clearChat);
  $("#chat-settings-toggle").addEventListener("click", () =>
    $("#chat-settings").classList.toggle("hidden"));
  const chatInput = $("#chat-input");
  chatInput.addEventListener("input", (e) => autoGrow(e.target));
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });

  // playground
  $("#pg-generate").addEventListener("click", runGenerate);
  $("#pg-stop").addEventListener("click", stopGenerate);

  // models
  $("#models-refresh").addEventListener("click", loadCheckpoints);
  $("#quant-run").addEventListener("click", runQuantize);

  // training
  $("#train-mode").addEventListener("change", onTrainModeChange);
  $("#train-preset").addEventListener("change", onPresetChange);
  $("#train-start").addEventListener("click", startTraining);
  $("#train-stop").addEventListener("click", stopTraining);
  window.addEventListener("resize", () => {
    if ($("#view-training").classList.contains("active")) drawChart();
  });

  refreshStatus();
  loadCheckpoints();
}

document.addEventListener("DOMContentLoaded", init);
