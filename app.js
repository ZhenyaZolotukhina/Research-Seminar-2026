// app.js
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

/**
 * Fully static, client-side sentiment analyzer:
 * - Loads reviews_test.tsv via fetch
 * - Parses TSV with Papa Parse (global Papa from index.html)
 * - Runs sentiment classification in-browser using Transformers.js pipeline
 * - Logs to Google Sheets via Apps Script Web App endpoint (optional)
 * - Visualizes confidence (donut) and session distribution (canvas)
 */

const MODEL_ID = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
const TSV_PATH = "reviews_test.tsv";

// ---- Minimal shared state (kept private in module scope) ----
let reviews = [];
let sentimentPipeline = null;

const sessionCounts = { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 };

const STORAGE_KEYS = {
  userId: "rsa_user_id",
  logEnabled: "rsa_log_enabled",
  logEndpoint: "rsa_log_endpoint",
};

// ---- DOM helpers ----
const $ = (id) => document.getElementById(id);

function setBusy(isBusy, message = "Working…") {
  const busy = $("busy");
  const btn = $("analyzeBtn");
  busy.style.display = isBusy ? "inline-flex" : "none";
  busy.querySelector("span")?.remove?.(); // no-op safety
  busy.childNodes.forEach(() => {}); // keep structure intact
  btn.disabled = isBusy || !sentimentPipeline || reviews.length === 0;
  if (isBusy) {
    busy.innerHTML = `<i class="fa-solid fa-spinner"></i> ${escapeHtml(message)}`;
  }
}

function setStatus(text, kind = "info") {
  const dot = $("statusDot");
  const statusText = $("statusText");

  statusText.textContent = text;

  dot.classList.remove("ready", "warn", "err");
  if (kind === "ready") dot.classList.add("ready");
  if (kind === "warn") dot.classList.add("warn");
  if (kind === "err") dot.classList.add("err");
}

function clearError() {
  const box = $("errorBox");
  box.textContent = "";
  box.classList.remove("show");
}

function showError(userMessage, err = null) {
  const box = $("errorBox");
  box.textContent = userMessage;
  box.classList.add("show");
  if (err) console.error(err);
}

function setReviewText(text) {
  const el = $("reviewText");
  el.textContent = text || "";
}

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// ---- Reviews TSV loading ----
async function loadReviews() {
  setStatus("Loading reviews TSV…", "info");
  try {
    const res = await fetch(endpoint, {
    method: "POST",
    mode: "no-cors",
    headers: { "Content-Type": "text/plain;charset=utf-8" },
    body: JSON.stringify(payload),
    cache: "no-store",
      });

    if (parsed.errors && parsed.errors.length > 0) {
      const first = parsed.errors[0];
      throw new Error(`TSV parse error: ${first.message || "Unknown"} (row ${first.row ?? "?"})`);
    }

    const rows = Array.isArray(parsed.data) ? parsed.data : [];
    const texts = rows
      .map((r) => (r && typeof r.text === "string" ? r.text.trim() : ""))
      .filter((t) => t.length > 0);

    if (texts.length === 0) {
      throw new Error(`No valid review texts found in TSV. Ensure it has a 'text' column.`);
    }

    reviews = texts;
    setStatus(`Reviews loaded (${reviews.length})`, "info");
  } catch (err) {
    reviews = [];
    setStatus("Reviews failed to load", "err");
    showError(
      `Could not load or parse ${TSV_PATH}. Make sure the file exists next to index.html and contains a 'text' column.`,
      err
    );
  }
}

// ---- Model init ----
async function initModel() {
  setStatus("Loading sentiment model… (first load can take a while)", "info");

  try {
    // Some versions support progress_callback; safe to try.
    sentimentPipeline = await pipeline("text-classification", MODEL_ID, {
      progress_callback: (p) => {
        if (!p) return;
        // p can be a string or object depending on implementation; keep it simple.
        const msg =
          typeof p === "string"
            ? p
            : (p?.status ? `${p.status}${p?.file ? `: ${p.file}` : ""}` : "Loading model…");
        setStatus(msg, "info");
      },
    });

    setStatus("Sentiment model ready", "ready");
  } catch (err) {
    sentimentPipeline = null;
    setStatus("Model failed to load", "err");
    showError(
      `Could not load the sentiment model in the browser. Check console for details and ensure your browser supports WebGPU/WebAssembly as needed.`,
      err
    );
  }
}

// ---- Sentiment analysis ----
function pickRandomReview() {
  if (!reviews.length) return null;
  const idx = Math.floor(Math.random() * reviews.length);
  return reviews[idx] || null;
}

function normalizePipelineOutput(output) {
  // Expected: [{label, score}, ...]
  if (Array.isArray(output) && output.length > 0 && output[0] && typeof output[0] === "object") {
    return output;
  }
  // Some pipelines return nested arrays; handle [[{...}]]
  if (Array.isArray(output) && output.length > 0 && Array.isArray(output[0]) && output[0][0]) {
    return output[0];
  }
  throw new Error("Unexpected pipeline output format.");
}

function mapToBucket(topLabel, topScore) {
  const label = String(topLabel || "").toUpperCase();
  const score = Number(topScore);

  if (label === "POSITIVE" && score > 0.5) return "POSITIVE";
  if (label === "NEGATIVE" && score > 0.5) return "NEGATIVE";
  return "NEUTRAL";
}

function bucketToUI(bucket) {
  if (bucket === "POSITIVE") return { icon: "fa-thumbs-up", cls: "accentPos" };
  if (bucket === "NEGATIVE") return { icon: "fa-thumbs-down", cls: "accentNeg" };
  return { icon: "fa-question-circle", cls: "accentNeu" };
}

function setResultUI({ bucket, modelLabel, score, ms }) {
  const badge = $("resultBadge");
  const iconWrap = $("resultIcon");
  const labelEl = $("resultLabel");
  const metaEl = $("resultMeta");

  const pct = Math.max(0, Math.min(1, Number(score) || 0)) * 100;
  $("confidencePct").textContent = `${pct.toFixed(1)}%`;

  // Donut arc update
  updateDonut(pct / 100, bucket);

  // Badge styling + icon
  const { icon, cls } = bucketToUI(bucket);
  badge.classList.remove("accentPos", "accentNeg", "accentNeu");
  badge.classList.add(cls);

  iconWrap.innerHTML = `<i class="fa-solid ${icon}"></i>`;

  const niceLabel = bucket;
  labelEl.textContent = `${niceLabel} (${pct.toFixed(1)}% confidence)`;

  const detail = `Model: ${MODEL_ID} • Raw label: ${String(modelLabel)} • ${(ms ?? 0)} ms`;
  metaEl.textContent = detail;
}

function updateDonut(progress01, bucket) {
  const arc = $("donutArc");
  const r = 48;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, progress01 || 0));
  const dash = clamped * circumference;
  const gap = circumference - dash;
  arc.setAttribute("stroke-dasharray", `${dash.toFixed(2)} ${gap.toFixed(2)}`);

  // Keep a subtle stroke color shift by sentiment via inline stroke with opacity (no hard-coded palette required)
  if (bucket === "POSITIVE") arc.setAttribute("stroke", "rgba(34,197,94,0.85)");
  else if (bucket === "NEGATIVE") arc.setAttribute("stroke", "rgba(239,68,68,0.85)");
  else arc.setAttribute("stroke", "rgba(163,163,163,0.75)");
}

async function analyzeRandomReview() {
  clearError();

  if (!sentimentPipeline) {
    showError("Sentiment model is not ready yet. Please wait for it to finish loading.");
    return;
  }
  if (!reviews.length) {
    showError("No reviews are loaded. Please check reviews_test.tsv and reload the page.");
    return;
  }

  const review = pickRandomReview();
  if (!review) {
    showError("Could not pick a review. Please check the TSV content.");
    return;
  }

  setReviewText(review);
  setBusy(true, "Analyzing…");

  const t0 = performance.now();
  try {
    const raw = await sentimentPipeline(review);
    const normalized = normalizePipelineOutput(raw);

    // Take top result
    const top = normalized
      .slice()
      .sort((a, b) => (Number(b?.score) || 0) - (Number(a?.score) || 0))[0];

    if (!top || typeof top.label !== "string" || typeof top.score !== "number") {
      throw new Error("Invalid top classification result.");
    }

    const bucket = mapToBucket(top.label, top.score);
    const dt = Math.round(performance.now() - t0);

    setResultUI({ bucket, modelLabel: top.label, score: top.score, ms: dt });

    // Update session distribution and chart
    sessionCounts[bucket] += 1;
    drawDistributionChart();
    updateChartFooter();

    // Optional: log to Google Sheet
    await maybeLogToSheet({
      ts_iso: new Date().toISOString(),
      review,
      sentiment: `${bucket} (${(top.score * 100).toFixed(1)}%)`,
      meta: buildMeta({ bucket, modelLabel: top.label, score: top.score, ms: dt }),
    });
  } catch (err) {
    showError("Analysis failed. Please try again (and check the console for details).", err);
  } finally {
    setBusy(false);
  }
}

// ---- Session chart (canvas) ----
function updateChartFooter() {
  $("chartFoot").textContent =
    `POSITIVE: ${sessionCounts.POSITIVE} • NEGATIVE: ${sessionCounts.NEGATIVE} • NEUTRAL: ${sessionCounts.NEUTRAL}`;
}

function drawDistributionChart() {
  const canvas = $("distChart");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // HiDPI scaling
  const cssWidth = canvas.clientWidth || 900;
  const cssHeight = canvas.clientHeight || 280;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssWidth * dpr);
  canvas.height = Math.floor(cssHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = cssWidth;
  const h = cssHeight;

  // Background
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(0,0,0,0.0)";
  ctx.fillRect(0, 0, w, h);

  // Values
  const labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"];
  const values = [sessionCounts.POSITIVE, sessionCounts.NEUTRAL, sessionCounts.NEGATIVE];
  const maxV = Math.max(1, ...values);

  const padding = { left: 28, right: 18, top: 16, bottom: 34 };
  const innerW = w - padding.left - padding.right;
  const innerH = h - padding.top - padding.bottom;

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.10)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (innerH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }

  const barGap = 22;
  const barW = (innerW - barGap * (labels.length - 1)) / labels.length;

  // Bars (subtle gradient per bar)
  labels.forEach((lab, i) => {
    const v = values[i];
    const x = padding.left + i * (barW + barGap);
    const barH = (v / maxV) * (innerH - 6);
    const y = padding.top + innerH - barH;

    const grad = ctx.createLinearGradient(0, y, 0, y + barH);
    if (lab === "POSITIVE") {
      grad.addColorStop(0, "rgba(34,197,94,0.75)");
      grad.addColorStop(1, "rgba(34,197,94,0.18)");
    } else if (lab === "NEGATIVE") {
      grad.addColorStop(0, "rgba(239,68,68,0.75)");
      grad.addColorStop(1, "rgba(239,68,68,0.18)");
    } else {
      grad.addColorStop(0, "rgba(163,163,163,0.65)");
      grad.addColorStop(1, "rgba(163,163,163,0.14)");
    }

    // Rounded rect bar
    roundRect(ctx, x, y, barW, barH, 12);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, barW, barH, 12);
    ctx.stroke();

    // Value label above
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "700 13px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(String(v), x + barW / 2, y - 6);

    // Category label below
    ctx.fillStyle = "rgba(255,255,255,0.65)";
    ctx.font = "650 12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    ctx.textBaseline = "top";
    ctx.fillText(lab, x + barW / 2, padding.top + innerH + 10);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

// ---- Logging to Google Sheet (optional) ----
function getOrCreateUserId() {
  const existing = localStorage.getItem(STORAGE_KEYS.userId);
  if (existing) return existing;

  // Simple random ID (not PII)
  const id = `u_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
  localStorage.setItem(STORAGE_KEYS.userId, id);
  return id;
}

function buildMeta(extra = {}) {
  const meta = {
    userId: getOrCreateUserId(),
    page: location.href,
    referrer: document.referrer || "",
    userAgent: navigator.userAgent,
    language: navigator.language || "",
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone || "",
    screen: { w: window.screen.width, h: window.screen.height, dpr: window.devicePixelRatio || 1 },
    model: MODEL_ID,
    app: { name: "review-sentiment-explorer", version: "1.0.0" },
    ...extra,
  };
  return meta;
}

function isLoggingEnabled() {
  return localStorage.getItem(STORAGE_KEYS.logEnabled) === "1";
}

function getLogEndpoint() {
  return (localStorage.getItem(STORAGE_KEYS.logEndpoint) || "").trim();
}

async function maybeLogToSheet({ ts_iso, review, sentiment, meta }) {
  if (!isLoggingEnabled()) return;
  const endpoint = getLogEndpoint();
  if (!endpoint) return;

  // Payload matches required columns (Timestamp, Review, Sentiment, Meta)
  const payload = {
    timestamp: ts_iso,
    review,
    sentiment,
    meta, // object; Apps Script can stringify it
  };

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      mode: "cors",
      cache: "no-store",
    });

    // Apps Script often returns 200 with a JSON body or plain text
    if (!res.ok) {
      throw new Error(`Logging failed: ${res.status} ${res.statusText}`);
    }
  } catch (err) {
    // Logging should never break UX; just warn and continue.
    console.warn("Google Sheet logging error:", err);
  }
}

// ---- UI: logging controls ----
function syncLoggingUIFromStorage() {
  const sw = $("logSwitch");
  const endpoint = $("logEndpoint");

  const enabled = isLoggingEnabled();
  sw.classList.toggle("on", enabled);
  sw.setAttribute("aria-checked", enabled ? "true" : "false");

  endpoint.value = getLogEndpoint();
}

function toggleLogging() {
  const enabled = !isLoggingEnabled();
  localStorage.setItem(STORAGE_KEYS.logEnabled, enabled ? "1" : "0");
  syncLoggingUIFromStorage();
}

function attachLoggingHandlers() {
  const sw = $("logSwitch");
  const endpoint = $("logEndpoint");

  sw.addEventListener("click", toggleLogging);
  sw.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleLogging();
    }
  });

  endpoint.addEventListener("change", () => {
    localStorage.setItem(STORAGE_KEYS.logEndpoint, endpoint.value.trim());
  });
}

// ---- Reset session ----
function resetSession() {
  sessionCounts.POSITIVE = 0;
  sessionCounts.NEGATIVE = 0;
  sessionCounts.NEUTRAL = 0;

  $("confidencePct").textContent = "—";
  updateDonut(0, "NEUTRAL");

  const badge = $("resultBadge");
  badge.classList.remove("accentPos", "accentNeg", "accentNeu");
  badge.classList.add("accentNeu");
  $("resultIcon").innerHTML = `<i class="fa-regular fa-circle-question"></i>`;
  $("resultLabel").textContent = "No result yet";
  $("resultMeta").textContent = "Click the button to run sentiment analysis.";

  setReviewText("Review will appear here after you click “Analyze random review”.");
  drawDistributionChart();
  updateChartFooter();
  clearError();
}

// ---- Bootstrap ----
function updateAnalyzeButtonState() {
  const btn = $("analyzeBtn");
  btn.disabled = !(sentimentPipeline && reviews.length > 0);
}

async function bootstrap() {
  attachLoggingHandlers();
  syncLoggingUIFromStorage();

  // Initial visuals
  updateDonut(0, "NEUTRAL");
  drawDistributionChart();
  updateChartFooter();

  $("analyzeBtn").addEventListener("click", analyzeRandomReview);
  $("resetBtn").addEventListener("click", resetSession);

  // Load reviews + model (sequential messages, but can be parallel)
  await loadReviews();
  await initModel();
  updateAnalyzeButtonState();

  // If one loaded but not the other, show a warning state
  if (reviews.length === 0 || !sentimentPipeline) {
    setStatus("Ready with issues (see error message)", "warn");
  }
}

document.addEventListener("DOMContentLoaded", bootstrap);
window.addEventListener("resize", () => {
  // Redraw chart on resize for crispness
  drawDistributionChart();
});
