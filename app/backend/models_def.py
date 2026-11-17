"""Definiciones de los modelos en PyTorch, idénticas a las del notebook.

Se usan para reconstruir la arquitectura y cargar los state_dict guardados en
`best.pt`. Mantener en sincronía con `models/train_models_kaggle.ipynb`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Modelo 1 — BiLSTM + atención (entrenado desde cero)
# =========================================================================
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.context(torch.tanh(self.proj(h))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return pooled, weights


class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden_dim=256,
                 num_classes=4, pad_idx=0, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )
        self.pad_idx = pad_idx

    def forward(self, x, lengths, return_attn=False):
        mask = (x != self.pad_idx).long()
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        pooled, weights = self.attn(out, mask)
        logits = self.fc(self.dropout(pooled))
        if return_attn:
            return logits, weights
        return logits


# =========================================================================
# Modelo 2 — TF-IDF + Regresión Softmax
# =========================================================================
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# =========================================================================
# Modelo 4 — NewsHybridNet (doble rama con fusión por compuerta)
# =========================================================================
class SpatialDropout1d(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.drop(x)
        return x.squeeze(-1).permute(0, 2, 1)


class MaskedAdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.context(torch.tanh(self.proj(h))).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1)
        pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        return pooled, w


class NewsHybridNet(nn.Module):
    def __init__(self, emb_matrix, num_classes=4, pad_idx=0, ngram_buckets=2 ** 17):
        super().__init__()
        vocab_size, emb_dim = emb_matrix.shape
        self.pad_idx = pad_idx

        # Rama A: semántica secuencial
        self.embedding = nn.Embedding.from_pretrained(
            emb_matrix, freeze=True, padding_idx=pad_idx)
        self.adapter = nn.Linear(emb_dim, 256)
        self.adapter_ln = nn.LayerNorm(256)
        self.spatial_drop = SpatialDropout1d(0.2)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=2, padding="same")
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding="same")
        self.conv5 = nn.Conv1d(256, 128, kernel_size=5, padding="same")
        self.cat_ln = nn.LayerNorm(384)
        self.bigru = nn.GRU(384, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = MaskedAdditiveAttention(256)

        # Rama B: léxica de subpalabras
        self.embbag = nn.EmbeddingBag(ngram_buckets, 64, mode="mean")

        # Fusión con compuerta
        self.proj_a = nn.Linear(512, 256)
        self.proj_b = nn.Linear(64, 256)
        self.gate = nn.Linear(512, 256)

        # Cabeza
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths, ngram_values, ngram_offsets, return_gate=False, return_attn=False):
        mask = (x != self.pad_idx).long()
        emb = self.embedding(x)
        a = F.gelu(self.adapter_ln(self.adapter(emb)))
        a = self.spatial_drop(a)
        a_t = a.transpose(1, 2)
        c = torch.cat([self.conv2(a_t), self.conv3(a_t), self.conv5(a_t)], dim=1)
        c = c.transpose(1, 2)
        c = F.relu(self.cat_ln(c))
        packed = nn.utils.rnn.pack_padded_sequence(
            c, lengths.cpu(), batch_first=True, enforce_sorted=False)
        g_out, _ = self.bigru(packed)
        g_out, _ = nn.utils.rnn.pad_packed_sequence(
            g_out, batch_first=True, total_length=x.size(1))
        att, attn_w = self.attn(g_out, mask)
        masked = g_out.masked_fill((mask == 0).unsqueeze(-1), float("-inf"))
        maxp, _ = masked.max(dim=1)
        maxp = torch.nan_to_num(maxp, neginf=0.0)
        h_a = torch.cat([att, maxp], dim=1)

        h_b = self.embbag(ngram_values, ngram_offsets)

        ha_p = self.proj_a(h_a)
        hb_p = self.proj_b(h_b)
        g = torch.sigmoid(self.gate(torch.cat([ha_p, hb_p], dim=1)))
        fused = g * ha_p + (1 - g) * hb_p

        logits = self.head(fused)
        out = [logits]
        if return_gate:
            out.append(g)
        if return_attn:
            out.append(attn_w)
        return out[0] if len(out) == 1 else tuple(out)
