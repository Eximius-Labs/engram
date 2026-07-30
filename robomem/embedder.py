"""The embedder protocol robomem builds against, plus small vector helpers.

robomem never constructs an embedder. It takes one by dependency injection: an object that
implements :class:`Embedder`. In production that object is
``fusion_embedding.unified.UnifiedEmbedder.from_pretrained(...)``; in tests it is
:class:`robomem.fakes.FakeEmbedder`, a deterministic CPU stand-in. This is the same
"DI at the seams + tiny CPU stand-ins" discipline the model core uses, and it is what keeps
the entire ingest -> index -> recall path runnable with no model, no GPU, and no network.

Every ``embed_*`` method returns a full-width (``VECTOR_DIM``), L2-normalized vector. Cross
-modal ranking goes through :meth:`Embedder.center` (per-modality mean-centering) because raw
cross-modal cosine carries a modality gap.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """The embed_* protocol. Any object implementing this can drive robomem.

    Return types are torch tensors in production (UnifiedEmbedder) but robomem coerces every
    vector to numpy float32 at the storage seam, so a stand-in may return a numpy array or a
    torch tensor. Vectors are expected full-width and L2-normalized.
    """

    def embed_text(self, text: str, instruction: Optional[str] = None, dim: Optional[int] = None): ...
    def embed_image(self, image, dim: Optional[int] = None): ...
    def embed_video(self, video, fps: Optional[float] = None,
                    max_frames: Optional[int] = None, dim: Optional[int] = None): ...
    def embed_audio(self, audio, sr: Optional[int] = None, dim: Optional[int] = None): ...
    def embed_thermal(self, image, dim: Optional[int] = None): ...
    def embed_motion(self, accel, dim: Optional[int] = None): ...
    def embed_geometry(self, image, dim: Optional[int] = None): ...
    def center(self, embs): ...
    def rank_cross_modal(self, queries, gallery, center: bool = True): ...


# --------------------------------------------------------------------------- #
# numpy helpers used across ingest and recall (framework-agnostic).
# --------------------------------------------------------------------------- #
def to_numpy(vec) -> np.ndarray:
    """Coerce a torch tensor / numpy array / sequence to a 1-D float32 numpy vector."""
    if hasattr(vec, "detach"):  # torch.Tensor
        vec = vec.detach().cpu().numpy()
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr


def l2_normalize(mat: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = to_numpy(a)
    b = to_numpy(b)
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
