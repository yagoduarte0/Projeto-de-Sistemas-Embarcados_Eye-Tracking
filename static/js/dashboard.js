// ── Socket ────────────────────────────────────────────────────────────────────
const socket = io();

// ── Estado ────────────────────────────────────────────────────────────────────
let sessionRunning = false;
let sessionStart = null;
let timerInterval = null;

// ── Gráfico de IAF ao longo da sessão ────────────────────────────────────────
const focusCtx = document.getElementById("focus-chart").getContext("2d");
const focusChart = new Chart(focusCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "IAF (%)",
      data: [],
      borderColor: "#1e90ff",
      backgroundColor: "rgba(30,144,255,.12)",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.35,
      fill: true,
    }],
  },
  options: {
    animation: false,
    responsive: true,
    scales: {
      x: { display: false },
      y: {
        min: 0, max: 100,
        ticks: { color: "#7b82a8", callback: v => v + "%" },
        grid: { color: "#2c3150" },
      },
    },
    plugins: { legend: { display: false } },
  },
});

let chartTick = 0;

function pushIAFPoint(iafPct) {
  chartTick++;
  focusChart.data.labels.push(chartTick);
  focusChart.data.datasets[0].data.push(iafPct);
  if (focusChart.data.labels.length > 180) {
    focusChart.data.labels.shift();
    focusChart.data.datasets[0].data.shift();
  }
  focusChart.update("none");
}

// ── Timer ─────────────────────────────────────────────────────────────────────
function formatTime(secs) {
  const m = String(Math.floor(secs / 60)).padStart(2, "0");
  const s = String(Math.floor(secs % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

function startTimer() {
  sessionStart = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = (Date.now() - sessionStart) / 1000;
    document.getElementById("session-timer").textContent = formatTime(elapsed);
  }, 500);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ── Controles de sessão ───────────────────────────────────────────────────────
async function startSession() {
  const res = await fetch("/api/start", { method: "POST" });
  if (!res.ok) { alert("Erro ao iniciar sessão"); return; }

  sessionRunning = true;
  document.getElementById("btn-start").classList.add("hidden");
  document.getElementById("btn-stop").classList.remove("hidden");
  document.getElementById("camera-placeholder").style.display = "none";
  document.getElementById("events-list").innerHTML = '<p class="empty-state">Aguardando eventos...</p>';
  chartTick = 0;
  focusChart.data.labels = [];
  focusChart.data.datasets[0].data = [];
  focusChart.update("none");
  setIAFGauge(null);
  startTimer();
}

async function stopSession() {
  const res = await fetch("/api/stop", { method: "POST" });
  const data = await res.json();

  sessionRunning = false;
  stopTimer();
  document.getElementById("btn-stop").classList.add("hidden");
  document.getElementById("btn-start").classList.remove("hidden");
  document.getElementById("camera-feed").src = "";
  document.getElementById("camera-placeholder").style.display = "flex";
  setFocusIndicator(null);
  setIAFGauge(null);
  if (data.stats) updateMetrics(data.stats);
}

// ── Exportar ──────────────────────────────────────────────────────────────────
function exportReport(format) {
  const a = document.createElement("a");
  a.href = `/api/export/${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ── Socket events ─────────────────────────────────────────────────────────────
socket.on("frame", ({ data, iaf }) => {
  if (!sessionRunning) return;
  document.getElementById("camera-feed").src = `data:image/jpeg;base64,${data}`;
  if (iaf != null) {
    setIAFGauge(iaf);
    pushIAFPoint(iaf);
  }
});

socket.on("stats_update", (stats) => {
  updateMetrics(stats);
});

socket.on("new_event", (ev) => {
  addEventItem(ev);
  const isDistracted = ["side_gaze", "distraction", "focus_lost"].includes(ev.kind);
  setFocusIndicator(isDistracted ? false : null);
});

socket.on("alert", ({ message }) => {
  showAlert(message);
  setFocusIndicator(false);
});

// ── UI helpers ────────────────────────────────────────────────────────────────
function updateMetrics(stats) {
  document.getElementById("m-focus").textContent =
    stats.focus_percentage != null ? `${stats.focus_percentage.toFixed(1)}%` : "—";
  document.getElementById("m-distractions").textContent = stats.total_distractions ?? 0;
  document.getElementById("m-side").textContent = stats.gaze_away_count ?? 0;
  document.getElementById("m-focus-lost").textContent = stats.focus_lost_count ?? 0;

  if (stats.iaf_mean != null) {
    const el = document.getElementById("m-iaf-mean");
    if (el) el.textContent = `${stats.iaf_mean.toFixed(1)}%`;
  }

  const focused = (stats.focus_percentage ?? 100) > 70;
  setFocusIndicator(focused);
}

function setFocusIndicator(focused) {
  const el = document.getElementById("focus-indicator");
  const label = document.getElementById("focus-label");
  el.classList.remove("focused", "distracted", "neutral");
  if (focused === null) {
    el.classList.add("neutral");
    label.textContent = "Aguardando...";
  } else if (focused) {
    el.classList.add("focused");
    label.textContent = "Focado";
  } else {
    el.classList.add("distracted");
    label.textContent = "Distraído!";
  }
}

function setIAFGauge(iafPct) {
  const bar  = document.getElementById("iaf-bar-fill");
  const val  = document.getElementById("iaf-gauge-value");
  if (!bar || !val) return;

  if (iafPct == null) {
    bar.style.width = "0%";
    bar.className = "iaf-bar-fill";
    val.textContent = "—";
    val.style.color = "var(--text-muted)";
    return;
  }

  bar.style.width = `${iafPct}%`;
  val.textContent = `${iafPct.toFixed(1)}%`;

  if (iafPct >= 70) {
    bar.className = "iaf-bar-fill high";
    val.style.color = "var(--success)";
  } else if (iafPct >= 40) {
    bar.className = "iaf-bar-fill mid";
    val.style.color = "var(--warning)";
  } else {
    bar.className = "iaf-bar-fill low";
    val.style.color = "var(--danger)";
  }
}

function addEventItem(ev) {
  const list = document.getElementById("events-list");
  const empty = list.querySelector(".empty-state");
  if (empty) empty.remove();

  const kindLabels = {
    side_gaze:  "Olhar evasivo",
    distraction: "Distração",
    focus_lost:  "Perda de foco",
    refocus:     "Refoco",
  };

  const item = document.createElement("div");
  item.className = `event-item ${ev.kind}`;
  item.innerHTML = `
    <span class="event-time">${ev.timestamp.toFixed(1)}s</span>
    <span class="event-badge ${ev.kind}">${kindLabels[ev.kind] ?? ev.kind}</span>
    <span class="event-detail">${ev.detail ?? ""}</span>
  `;
  list.prepend(item);
  while (list.children.length > 50) list.removeChild(list.lastChild);
}

let alertTimeout = null;
function showAlert(message) {
  const banner = document.getElementById("alert-banner");
  document.getElementById("alert-text").textContent = message;
  banner.classList.remove("hidden");
  clearTimeout(alertTimeout);
  alertTimeout = setTimeout(() => banner.classList.add("hidden"), 5000);
}
