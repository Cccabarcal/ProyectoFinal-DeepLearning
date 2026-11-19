"""Preprocesamiento compartido, idéntico al del notebook de entrenamiento.

Replica exactamente la tokenización por palabras, la codificación a índices y
los n-gramas de caracteres con hashing, para que la inferencia reproduzca el
preprocesamiento con el que se entrenaron los modelos.
"""
from __future__ import annotations

import re
import zlib
from typing import List, Tuple

PAD, UNK = "<pad>", "<unk>"

# --- Tokenización por palabras (modelos RNN y NewsHybridNet) ---
_token_re = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())


def encode(text: str, stoi: dict, max_len: int, pad_idx: int) -> Tuple[List[int], int]:
    unk = stoi.get(UNK, 1)
    ids = [stoi.get(tok, unk) for tok in tokenize(text)][:max_len]
    length = len(ids)
    if length < max_len:
        ids = ids + [pad_idx] * (max_len - length)
    return ids, max(length, 1)


# --- N-gramas de caracteres con hashing (rama B de NewsHybridNet) ---
_norm_re = re.compile(r"\s+")


def _normalize_for_ngrams(text: str) -> str:
    return _norm_re.sub(" ", text.lower()).strip()


def char_ngram_hashes(text: str, buckets: int, ns: Tuple[int, ...]) -> List[int]:
    s = "<" + _normalize_for_ngrams(text) + ">"
    idxs: List[int] = []
    for n in ns:
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            gram = s[i:i + n]
            idxs.append(zlib.crc32(gram.encode("utf-8")) % buckets)
    if not idxs:
        idxs = [0]
    return idxs
