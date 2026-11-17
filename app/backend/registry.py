"""Registro y carga perezosa de los cuatro modelos.

Cada modelo se reconstruye desde `models/checkpoints/<modelo>/` reproduciendo el
preprocesamiento del notebook. Si faltan los pesos (todavía no se han entrenado),
el modelo pasa a MODO DEMO: devuelve una predicción simulada y determinista a
partir de palabras clave, de modo que el frontend se puede desarrollar y probar
sin esperar al entrenamiento. La respuesta siempre indica `demo: true/false`.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from preprocessing import PAD, UNK, char_ngram_hashes, encode, tokenize

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_CLASSES = len(CLASS_NAMES)

# Raíz de checkpoints: <repo>/models/checkpoints
BACKEND_DIR = Path(__file__).resolve().parent
REPO_DIR = BACKEND_DIR.parent.parent           # ProyectoFinal-DeepLearning/
CKPT_ROOT = REPO_DIR / "models" / "checkpoints"

# Pistas léxicas para el modo demo (solo cuando no hay pesos entrenados).
_DEMO_KEYWORDS = {
    "World": ["president", "government", "war", "country", "minister", "election",
              "nations", "military", "attack", "peace", "border", "iraq", "israel"],
    "Sports": ["game", "match", "team", "win", "season", "coach", "league",
               "championship", "player", "goal", "cup", "olympic", "football"],
    "Business": ["market", "stocks", "company", "profit", "economy", "dollar",
                 "shares", "bank", "oil", "prices", "trade", "sales", "quarterly"],
    "Sci/Tech": ["software", "internet", "computer", "google", "chip", "space",
                 "research", "technology", "data", "ai", "phone", "science", "nasa"],
}

_torch = None


def _lazy_torch():
    global _torch
    if _torch is None:
        import torch  # import tardío: el modo demo no necesita torch
        _torch = torch
    return _torch


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class BaseModel:
    """Contrato común. `key` es el id estable usado por el frontend/API."""

    key: str
    name: str
    description: str

    def __init__(self):
        self.loaded = False
        self.demo = True
        self.load_error: Optional[str] = None
        self._net = None

    @property
    def ckpt_dir(self) -> Path:
        return CKPT_ROOT / self.key

    def available(self) -> bool:
        """¿Existen los artefactos mínimos para cargar pesos reales?"""
        return (self.ckpt_dir / "best.pt").exists()

    def ensure_loaded(self):
        if self.loaded:
            return
        if self.available():
            try:
                self._load_real()
                self.demo = False
            except Exception as e:  # noqa: BLE001 - degradar a demo sin romper la API
                self.load_error = f"{type(e).__name__}: {e}"
                self.demo = True
        self.loaded = True

    def _load_real(self):
        raise NotImplementedError

    def _predict_real(self, text: str) -> Dict:
        raise NotImplementedError

    # ---- Modo demo: predicción determinista por palabras clave ----
    def _predict_demo(self, text: str) -> Dict:
        toks = tokenize(text)
        scores = np.zeros(NUM_CLASSES, dtype=np.float64)
        for i, cls in enumerate(CLASS_NAMES):
            kws = set(_DEMO_KEYWORDS[cls])
            scores[i] = sum(1.0 for t in toks if t in kws)
        # Pequeño sesgo determinista por hash para romper empates de forma estable.
        h = int(hashlib.md5(text.strip().lower().encode()).hexdigest(), 16)
        for i in range(NUM_CLASSES):
            scores[i] += ((h >> (i * 8)) & 0xFF) / 2550.0
        if scores.sum() == 0:
            scores += 0.25
        probs = _softmax(scores * 1.5)
        idx = int(np.argmax(probs))
        return {
            "label": CLASS_NAMES[idx],
            "label_index": idx,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }

    def predict(self, text: str) -> Dict:
        self.ensure_loaded()
        t0 = time.perf_counter()
        if self.demo:
            result = self._predict_demo(text)
        else:
            result = self._predict_real(text)
        result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["demo"] = self.demo
        return result

    def info(self) -> Dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "available": self.available(),
            "load_error": self.load_error,
        }


# =========================================================================
class RnnModel(BaseModel):
    key = "rnn_scratch"
    name = "BiLSTM + Atención"
    description = "Red recurrente bidireccional con atención, entrenada desde cero."

    def _load_real(self):
        torch = _lazy_torch()
        from models_def import BiLSTMAttentionClassifier
        vocab = json.load(open(self.ckpt_dir / "vocab.json"))
        self.itos = vocab["itos"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.max_len = vocab["max_len"]
        self.pad_idx = vocab["pad_idx"]
        net = BiLSTMAttentionClassifier(len(self.itos), pad_idx=self.pad_idx)
        best = torch.load(self.ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(best["model_state"])
        net.eval()
        self._net = net

    def _predict_real(self, text: str) -> Dict:
        torch = _lazy_torch()
        ids, length = encode(text, self.stoi, self.max_len, self.pad_idx)
        x = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        with torch.no_grad():
            logits, attn = self._net(x, lengths, return_attn=True)
        probs = _softmax(logits[0].numpy())
        idx = int(np.argmax(probs))
        toks = tokenize(text)[: self.max_len]
        weights = attn[0][: len(toks)].tolist()
        return {
            "label": CLASS_NAMES[idx],
            "label_index": idx,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            "attention": [{"token": t, "weight": float(w)} for t, w in zip(toks, weights)],
        }


# =========================================================================
class TfidfModel(BaseModel):
    key = "tfidf_softmax"
    name = "TF-IDF + Softmax"
    description = "Línea base clásica: TF-IDF (uni+bigramas) con regresión softmax."

    def _load_real(self):
        import joblib
        torch = _lazy_torch()
        from models_def import SoftmaxRegression
        self.vectorizer = joblib.load(self.ckpt_dir / "tfidf_vectorizer.joblib")
        input_dim = len(self.vectorizer.get_feature_names_out())
        net = SoftmaxRegression(input_dim)
        best = torch.load(self.ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(best["model_state"])
        net.eval()
        self._net = net

    def _predict_real(self, text: str) -> Dict:
        torch = _lazy_torch()
        vec = self.vectorizer.transform([text])
        x = torch.from_numpy(vec.toarray().astype(np.float32))
        with torch.no_grad():
            logits = self._net(x)
        probs = _softmax(logits[0].numpy())
        idx = int(np.argmax(probs))
        return {
            "label": CLASS_NAMES[idx],
            "label_index": idx,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }


# =========================================================================
class DistilBertModel(BaseModel):
    key = "distilbert"
    name = "DistilBERT (fine-tuning)"
    description = "Transformer preentrenado ajustado por completo sobre AG News."

    def available(self) -> bool:
        return (self.ckpt_dir / "hf_best").exists() and (self.ckpt_dir / "tokenizer").exists()

    def _load_real(self):
        torch = _lazy_torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.ckpt_dir / "tokenizer"))
        self._net = AutoModelForSequenceClassification.from_pretrained(str(self.ckpt_dir / "hf_best"))
        self._net.eval()
        self.max_tokens = 128

    def _predict_real(self, text: str) -> Dict:
        torch = _lazy_torch()
        enc = self.tokenizer(text, truncation=True, max_length=self.max_tokens,
                             padding="max_length", return_tensors="pt")
        with torch.no_grad():
            out = self._net(**enc)
        probs = _softmax(out.logits[0].numpy())
        idx = int(np.argmax(probs))
        return {
            "label": CLASS_NAMES[idx],
            "label_index": idx,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        }


# =========================================================================
class HybridModel(BaseModel):
    key = "newshybridnet"
    name = "NewsHybridNet"
    description = "Arquitectura propia de doble rama (semántica + léxica) con fusión por compuerta."

    def available(self) -> bool:
        need = ["best.pt", "vocab.json", "embeddings.npy", "ngram_config.json"]
        return all((self.ckpt_dir / f).exists() for f in need)

    def _predict_demo(self, text: str) -> Dict:
        # Reutiliza la predicción base y añade atención + compuerta simuladas
        # para que la UI de NewsHybridNet sea visible antes de tener pesos.
        base = super()._predict_demo(text)
        toks = tokenize(text)[:40]
        kws = {w for kw in _DEMO_KEYWORDS.values() for w in kw}
        raw = []
        for t in toks:
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            w = 1.0 + 2.5 * (1.0 if t in kws else 0.0) + (h % 100) / 400.0
            raw.append(w)
        s = sum(raw) or 1.0
        base["attention"] = [{"token": t, "weight": w / s} for t, w in zip(toks, raw)]
        n_known = sum(1 for t in toks if t in kws)
        semantic = 0.45 + min(0.4, 0.06 * n_known)
        base["gate"] = {"semantic": semantic, "lexical": 1.0 - semantic}
        return base

    def _load_real(self):
        torch = _lazy_torch()
        from models_def import NewsHybridNet
        vocab = json.load(open(self.ckpt_dir / "vocab.json"))
        self.itos = vocab["itos"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.max_len = vocab["max_len"]
        self.pad_idx = vocab["pad_idx"]
        ng = json.load(open(self.ckpt_dir / "ngram_config.json"))
        self.buckets = ng["buckets"]
        self.ns = tuple(ng["ns"])
        emb = np.load(self.ckpt_dir / "embeddings.npy")
        net = NewsHybridNet(torch.tensor(emb), pad_idx=self.pad_idx, ngram_buckets=self.buckets)
        best = torch.load(self.ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(best["model_state"])
        net.eval()
        self._net = net

    def _predict_real(self, text: str) -> Dict:
        torch = _lazy_torch()
        ids, length = encode(text, self.stoi, self.max_len, self.pad_idx)
        ngrams = char_ngram_hashes(text, self.buckets, self.ns)
        x = torch.tensor([ids], dtype=torch.long)
        lengths = torch.tensor([length], dtype=torch.long)
        vals = torch.tensor(ngrams, dtype=torch.long)
        offs = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            logits, gate, attn = self._net(x, lengths, vals, offs,
                                           return_gate=True, return_attn=True)
        probs = _softmax(logits[0].numpy())
        idx = int(np.argmax(probs))
        toks = tokenize(text)[: self.max_len]
        weights = attn[0][: len(toks)].tolist()
        gate_mean = float(gate[0].mean())  # peso medio de la rama semántica (A)
        return {
            "label": CLASS_NAMES[idx],
            "label_index": idx,
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
            "attention": [{"token": t, "weight": float(w)} for t, w in zip(toks, weights)],
            "gate": {"semantic": gate_mean, "lexical": 1.0 - gate_mean},
        }


class Registry:
    def __init__(self):
        self.models: List[BaseModel] = [
            TfidfModel(), RnnModel(), HybridModel(), DistilBertModel(),
        ]
        self._by_key = {m.key: m for m in self.models}

    def get(self, key: str) -> Optional[BaseModel]:
        return self._by_key.get(key)

    def list_info(self) -> List[Dict]:
        return [m.info() for m in self.models]

    def predict_all(self, text: str) -> List[Dict]:
        results = []
        for m in self.models:
            r = m.predict(text)
            r.update({"key": m.key, "name": m.name})
            results.append(r)
        return results


REGISTRY = Registry()
