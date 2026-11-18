/* Lógica del frontend del clasificador de noticias. */

const API = {
  models: "/api/models",
  predict: "/api/predict",
};

const CLASS_COLOR_CLASS = {
  "World": "cls-World",
  "Sports": "cls-Sports",
  "Business": "cls-Business",
  "Sci/Tech": "cls-SciTech",
};

const el = (id) => document.getElementById(id);

const input = el("news-input");
const classifyBtn = el("classify-btn");
const clearBtn = el("clear-btn");
const charcount = el("charcount");
const resultsEl = el("results");
const statusEl = el("status");
const demoBadge = el("demo-badge");
const attentionSection = el("attention-section");
const themeToggle = el("theme-toggle");

let classes = ["World", "Sports", "Business", "Sci/Tech"];

/* ---------- Tema ---------- */
function initTheme() {
  const saved = localStorage.getItem("theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
  themeToggle.textContent = saved === "dark" ? "🌙" : "☀️";
}
themeToggle.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  themeToggle.textContent = next === "dark" ? "🌙" : "☀️";
  localStorage.setItem("theme", next);
});

/* ---------- Utilidades UI ---------- */
function setStatus(msg, kind = "info") {
  if (!msg) { statusEl.hidden = true; return; }
  statusEl.hidden = false;
  statusEl.className = `status ${kind}`;
  statusEl.textContent = msg;
}

function setLoading(on) {
  const label = classifyBtn.querySelector(".btn-label");
  const spinner = classifyBtn.querySelector(".spinner");
  classifyBtn.disabled = on;
  label.textContent = on ? "Clasificando" : "Clasificar";
  spinner.hidden = !on;
}

function updateCharcount() {
  charcount.textContent = input.value.length;
}

/* ---------- Carga inicial de modelos ---------- */
async function loadModels() {
  try {
    const res = await fetch(API.models);
    const data = await res.json();
    classes = data.classes || classes;
    const anyDemo = data.models.some((m) => !m.available);
    demoBadge.hidden = !anyDemo;
    if (anyDemo) {
      setStatus(
        "Algunos modelos aún no tienen pesos entrenados: se muestran predicciones en modo demo (simuladas). " +
        "En cuanto se coloquen los checkpoints en models/checkpoints/, la app usará los modelos reales automáticamente.",
        "info"
      );
    }
  } catch (e) {
    setStatus("No se pudo contactar con la API. ¿Está corriendo el servidor?", "error");
  }
}

/* ---------- Render de resultados ---------- */
function clsColorClass(label) {
  return CLASS_COLOR_CLASS[label] || "cls-World";
}

function renderResults(payload) {
  resultsEl.innerHTML = "";
  const template = el("result-card-template");

  payload.results.forEach((r) => {
    const node = template.content.cloneNode(true);
    node.querySelector(".model-name").textContent = r.name;
    node.querySelector(".model-desc").textContent = describe(r.key);

    const modeBadge = node.querySelector(".badge-mode");
    modeBadge.textContent = r.demo ? "demo" : "real";
    modeBadge.classList.add(r.demo ? "is-demo" : "is-real");

    const chip = node.querySelector(".pred-label-chip");
    chip.textContent = r.label;
    chip.classList.add(clsColorClass(r.label));

    const topProb = r.probs[r.label];
    node.querySelector(".pred-conf").textContent = `${(topProb * 100).toFixed(1)}% de confianza`;

    const bars = node.querySelector(".bars");
    const ordered = [...classes].sort((a, b) => r.probs[b] - r.probs[a]);
    ordered.forEach((cls) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span class="bar-label">${cls}</span>
        <div class="bar-track"><div class="bar-fill ${clsColorClass(cls)}"></div></div>
        <span class="bar-pct">${(r.probs[cls] * 100).toFixed(1)}%</span>`;
      bars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".bar-fill").style.width = `${r.probs[cls] * 100}%`;
      });
    });

    node.querySelector(".latency").textContent = `${r.inference_ms} ms`;
    resultsEl.appendChild(node);
  });

  renderAttention(payload);
}

function describe(key) {
  const map = {
    tfidf_softmax: "Línea base clásica: TF-IDF + regresión softmax.",
    rnn_scratch: "BiLSTM bidireccional con atención, desde cero.",
    newshybridnet: "Doble rama (semántica + léxica) con fusión por compuerta.",
    distilbert: "Transformer preentrenado y ajustado (fine-tuning).",
  };
  return map[key] || "";
}

/* ---------- Atención + compuerta de NewsHybridNet ---------- */
function renderAttention(payload) {
  const hybrid = payload.results.find((r) => r.key === "newshybridnet");
  if (!hybrid || !hybrid.attention || hybrid.attention.length === 0) {
    attentionSection.hidden = true;
    return;
  }
  attentionSection.hidden = false;

  const tokensEl = el("attention-tokens");
  tokensEl.innerHTML = "";
  const maxW = Math.max(...hybrid.attention.map((a) => a.weight), 1e-6);
  hybrid.attention.forEach((a) => {
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = a.token;
    const intensity = a.weight / maxW;
    span.style.background = `rgba(99, 102, 241, ${(0.08 + intensity * 0.7).toFixed(3)})`;
    span.title = `peso ${(a.weight * 100).toFixed(1)}%`;
    tokensEl.appendChild(span);
  });

  if (hybrid.gate) {
    const sem = Math.round(hybrid.gate.semantic * 100);
    el("gate-semantic").style.width = `${sem}%`;
    el("gate-lexical").style.width = `${100 - sem}%`;
    el("gate-semantic").querySelector("span").textContent = `Semántica ${sem}%`;
    el("gate-lexical").querySelector("span").textContent = `Léxica ${100 - sem}%`;
  }
}

/* ---------- Acción principal ---------- */
async function classify() {
  const text = input.value.trim();
  if (!text) {
    setStatus("Escribe o pega una noticia primero.", "error");
    input.focus();
    return;
  }
  setLoading(true);
  try {
    const res = await fetch(API.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    const data = await res.json();
    const anyDemo = data.results.some((r) => r.demo);
    if (!anyDemo) setStatus("");
    renderResults(data);
    resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    setStatus(`No se pudo clasificar: ${e.message}`, "error");
  } finally {
    setLoading(false);
  }
}

/* ---------- Eventos ---------- */
classifyBtn.addEventListener("click", classify);
clearBtn.addEventListener("click", () => {
  input.value = "";
  updateCharcount();
  resultsEl.innerHTML = "";
  attentionSection.hidden = true;
  input.focus();
});
input.addEventListener("input", updateCharcount);
input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") classify();
});
document.getElementById("examples").addEventListener("click", (e) => {
  const btn = e.target.closest(".chip");
  if (!btn) return;
  input.value = btn.dataset.text;
  updateCharcount();
  classify();
});

/* ============================================================
   Presentación embebida (deck tipo PowerPoint)
   ============================================================ */
const deck = el("deck");
const deckTrack = el("deck-track");
const slidePrev = el("slide-prev");
const slideNext = el("slide-next");
const slideFs = el("slide-fs");
const slideIndicator = el("slide-indicator");

if (deck && deckTrack) {
  const slides = Array.from(deckTrack.querySelectorAll(".slide"));
  let current = 0;

  function renderSlide() {
    // Oculta todas las slides y muestra solo la activa.
    slides.forEach((s, i) => {
      s.classList.toggle("is-active", i === current);
      s.setAttribute("aria-hidden", i === current ? "false" : "true");
    });
    slideIndicator.textContent = `${current + 1} / ${slides.length}`;
    slidePrev.disabled = current === 0;
    slideNext.disabled = current === slides.length - 1;
    // Reinicia el scroll de la slide visible.
    if (slides[current]) slides[current].scrollTop = 0;
  }

  function goTo(i) {
    current = Math.max(0, Math.min(slides.length - 1, i));
    renderSlide();
  }
  const nextSlide = () => goTo(current + 1);
  const prevSlide = () => goTo(current - 1);

  slideNext.addEventListener("click", nextSlide);
  slidePrev.addEventListener("click", prevSlide);

  /* Teclado: ← → PageUp/PageDown Home/End.
     Se ignora si el foco está en el textarea (salvo en pantalla completa). */
  document.addEventListener("keydown", (e) => {
    const typing = document.activeElement === input;
    const fsOnDeck = (document.fullscreenElement || document.webkitFullscreenElement) === deck;
    if (typing && !fsOnDeck) return;
    if (e.key === "ArrowRight" || e.key === "PageDown") { nextSlide(); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { prevSlide(); }
    else if (e.key === "Home") { goTo(0); }
    else if (e.key === "End") { goTo(slides.length - 1); }
  });

  /* Pantalla completa (Fullscreen API con fallback -webkit) */
  function isFullscreen() {
    return document.fullscreenElement || document.webkitFullscreenElement;
  }
  async function toggleFullscreen() {
    try {
      if (!isFullscreen()) {
        if (deck.requestFullscreen) await deck.requestFullscreen();
        else if (deck.webkitRequestFullscreen) deck.webkitRequestFullscreen();
      } else {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      }
    } catch (err) {
      setStatus(`No se pudo cambiar a pantalla completa: ${err.message}`, "error");
    }
  }
  function syncFsButton() {
    const on = !!isFullscreen();
    slideFs.textContent = on ? "⤢ Salir de pantalla completa" : "⛶ Pantalla completa";
  }
  slideFs.addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", syncFsButton);
  document.addEventListener("webkitfullscreenchange", syncFsButton);

  renderSlide();
}

/* ---------- Init ---------- */
initTheme();
updateCharcount();
loadModels();
