"""Shared fixtures: a deterministic fake embedder and a small synthetic session.

No real media files are needed. The fake embedder derives its semantic vector from the media
*path string* (filename concept tokens), so the whole ingest -> index -> recall path runs on
CPU with nothing on disk but the LanceDB store.
"""

import pytest

from robomem import FakeEmbedder, RobotMemory


def session_events():
    """A couple of events per modality, over ~6 seconds, with dog/cat concepts planted.

    ``img_dog2`` is a deliberate near-duplicate of ``img_dog1`` (same concept, same camera,
    0.4 s later) to exercise embedding dedup.
    """
    return [
        {"id": "img_dog1", "t": 1.0, "modality": "image",
         "path_or_data": "cam0/dog_01.png", "source": "cam0", "meta": {"path": "cam0/dog_01.png"}},
        {"id": "img_dog2", "t": 1.4, "modality": "image",
         "path_or_data": "cam0/dog_02.png", "source": "cam0", "meta": {"path": "cam0/dog_02.png"}},
        {"id": "img_cat", "t": 5.0, "modality": "image",
         "path_or_data": "cam0/cat_01.png", "source": "cam0", "meta": {"path": "cam0/cat_01.png"}},
        {"id": "aud_dog", "t": 1.2, "duration": 0.5, "modality": "audio",
         "path_or_data": "mic0/dog_bark.wav", "source": "mic0", "meta": {"path": "mic0/dog_bark.wav"}},
        {"id": "aud_cat", "t": 5.2, "duration": 0.5, "modality": "audio",
         "path_or_data": "mic0/cat_meow.wav", "source": "mic0", "meta": {"path": "mic0/cat_meow.wav"}},
        {"id": "txt_dog", "t": 2.0, "modality": "text",
         "path_or_data": "a dog runs across the yard", "source": "log"},
        {"id": "txt_cat", "t": 6.0, "modality": "text",
         "path_or_data": "a cat sleeps on the couch", "source": "log"},
        {"id": "mot0", "t": 3.0, "duration": 1.0, "modality": "motion",
         "path_or_data": {"data": [[0.1, 0.2, 9.8], [0.1, 0.2, 9.7]]}, "source": "imu0"},
        {"id": "thm_dog", "t": 3.5, "modality": "thermal",
         "path_or_data": "ir0/dog_ir.png", "source": "ir0", "meta": {"path": "ir0/dog_ir.png"}},
    ]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def make_mem(tmp_path, embedder):
    """Factory: open a fresh store and optionally index the synthetic session."""
    def _make(index=True, dedup_tau=None, name="db"):
        mem = RobotMemory.open(str(tmp_path / name), embedder=embedder)
        stats = None
        if index:
            stats = mem.index(session_events(), dedup_tau=dedup_tau)
        return mem, stats
    return _make


@pytest.fixture
def mem(make_mem):
    m, _ = make_mem()
    return m
