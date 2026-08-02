"""A deterministic, GPU-free stand-in embedder for tests and offline demos.

:class:`FakeEmbedder` implements the full ``embed_*`` protocol (see ``robomem.embedder``)
without any model, GPU, or network. It maps an input to a bag-of-concept-tokens semantic
vector, so two inputs that share a concept (a ``"dog"`` text query and a ``dog_01.png`` image)
land near each other, while unrelated concepts are near-orthogonal. A small fixed per-modality
offset is added so the modality gap is real and per-modality centering has a measurable effect,
exactly as it does with the trained model.

This is the ``fusion_embedding._tiny`` idea applied at the memory layer: keep the whole
ingest -> index -> recall path runnable on CPU, and swap in the real embedder for production.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import numpy as np

from .embedder import l2_normalize
from .schema import VECTOR_DIM

_MEDIA_EXT = {"png", "jpg", "jpeg", "bmp", "tiff", "gif", "wav", "flac", "mp3",
              "ogg", "mp4", "avi", "mov", "npy", "bin"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:8], "big")


class FakeEmbedder:
    """Deterministic unit-vector embedder. ``alpha`` sets the per-modality offset strength."""

    def __init__(self, dim: int = VECTOR_DIM, alpha: float = 0.3):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self._tok_cache: dict[str, np.ndarray] = {}
        self._offsets: dict[str, np.ndarray] = {}

    # ------------------------------------------------------- concept tokens
    def _token_vec(self, tok: str) -> np.ndarray:
        v = self._tok_cache.get(tok)
        if v is None:
            rng = np.random.RandomState(_seed("tok::" + tok) % (2**32))
            v = l2_normalize(rng.normal(size=self.dim).astype(np.float32))
            self._tok_cache[tok] = v
        return v

    def _modality_offset(self, modality: str) -> np.ndarray:
        v = self._offsets.get(modality)
        if v is None:
            rng = np.random.RandomState(_seed("mod::" + modality) % (2**32))
            v = l2_normalize(rng.normal(size=self.dim).astype(np.float32))
            self._offsets[modality] = v
        return v

    def _concepts(self, x) -> list[str]:
        """Extract concept tokens from an arbitrary input (path, text, array, dict, image)."""
        if isinstance(x, dict):
            if "concept" in x:
                return _TOKEN_RE.findall(str(x["concept"]).lower())
            x = x.get("data", x)
        if isinstance(x, (str, os.PathLike)):
            s = str(x)
            stem = os.path.splitext(os.path.basename(s))[0] if ("/" in s or "\\" in s or "." in s) else s
            toks = [t for t in _TOKEN_RE.findall(stem.lower())
                    if not t.isdigit() and t not in _MEDIA_EXT]
            return toks or ["_" + hashlib.md5(s.encode()).hexdigest()[:6]]
        # arrays / tensors / PIL images -> a single stable pseudo-token from their bytes
        try:
            if hasattr(x, "tobytes"):
                raw = x.tobytes()
            else:
                raw = np.asarray(x).tobytes()
        except Exception:
            raw = repr(x).encode("utf-8")
        return ["#" + hashlib.md5(raw).hexdigest()[:8]]

    def _semantic(self, x) -> np.ndarray:
        toks = self._concepts(x)
        acc = np.sum([self._token_vec(t) for t in toks], axis=0)
        return l2_normalize(acc.astype(np.float32))

    def _emit(self, x, modality: str, dim: Optional[int]) -> np.ndarray:
        v = self._semantic(x) + self.alpha * self._modality_offset(modality)
        v = l2_normalize(v)
        d = int(dim) if dim else self.dim
        return l2_normalize(v[:d])

    # ------------------------------------------------------- embed_* protocol
    def embed_text(self, text, instruction=None, dim=None):
        return self._emit(text, "text", dim)

    def embed_image(self, image, dim=None):
        return self._emit(image, "image", dim)

    def embed_video(self, video, fps=None, max_frames=None, dim=None):
        return self._emit(video, "video", dim)

    def embed_audio(self, audio, sr=None, dim=None):
        return self._emit(audio, "audio", dim)

    def embed_thermal(self, image, dim=None):
        return self._emit(image, "thermal", dim)

    def embed_motion(self, accel, dim=None):
        return self._emit(accel, "motion", dim)

    def embed_tactile(self, pressure, dim=None):
        return self._emit(pressure, "tactile", dim)

    def embed_geometry(self, image, dim=None):
        return self._emit(image, "geometry", dim)

    # -------------------------------------------- cross-modal readout helpers
    @staticmethod
    def center(embs):
        embs = np.asarray(embs, dtype=np.float32)
        return l2_normalize(embs - embs.mean(axis=0, keepdims=True))

    @staticmethod
    def rank_cross_modal(queries, gallery, center: bool = True):
        q = FakeEmbedder.center(queries) if center else np.asarray(queries, np.float32)
        g = FakeEmbedder.center(gallery) if center else np.asarray(gallery, np.float32)
        return q @ g.T
