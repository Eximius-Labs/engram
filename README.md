# robomem

The robot-memory layer for Eximius Labs. `robomem` sits on top of the unified multimodal
embedding space: it ingests a recorded multimodal session, indexes it, and answers
natural-language recall queries over it.

It is a plugin, not part of the model core. It lives in its own repository (heavy robotics
dependencies such as LanceDB, and a license firewall separate from the permissive model core)
and depends on the `fusion_embedding` package only through a small injected interface.

## What it does

Give it a session (a list of timestamped media events from a robot's sensors) and it will:

1. **Ingest** the session, group events by modality, embed each with the unified embedder,
   and write one row per window into a local LanceDB table.
2. **Recall** windows by natural language: `mem.recall("the person who dropped a mug")`.
3. **Recall by example** across modalities: hand it any stored vector and ask for the nearest
   windows of another modality, with no caption round-trip.

The current MVP is the **search** phase (Phase C). The temporal / episodic reasoning phase
(Phase E) is deferred; the schema already reserves its columns.

## Design: dependency injection at the embedder seam

`robomem` never builds an embedder. It takes one that implements the `embed_*` protocol
(`robomem.embedder.Embedder`):

```
embed_text  embed_image  embed_video  embed_audio  embed_thermal  embed_motion  embed_geometry
center      rank_cross_modal
```

Every `embed_*` returns a full-width, L2-normalized vector. In production that object is the
trained `UnifiedEmbedder`; in tests it is `FakeEmbedder`, a deterministic CPU stand-in. This
keeps the whole ingest -> index -> recall path runnable with no model, no GPU, and no network,
following the model core's own "DI at the seams plus tiny CPU stand-ins" testing discipline.

## Quickstart (CPU, no model)

```python
from robomem import RobotMemory, FakeEmbedder

session = [
    {"id": "img_dog", "t": 1.0, "modality": "image",
     "path_or_data": "cam0/dog_01.png", "source": "cam0"},
    {"id": "aud_dog", "t": 1.2, "duration": 0.5, "modality": "audio",
     "path_or_data": "mic0/dog_bark.wav", "source": "mic0"},
    {"id": "txt", "t": 2.0, "modality": "text",
     "path_or_data": "a dog runs across the yard", "source": "log"},
]

mem = RobotMemory.open("./session.lancedb", embedder=FakeEmbedder())
mem.index(session, dedup_tau=0.98)          # dedup_tau is optional (edge-storage lever)

hits = mem.recall("dog", modality="image", k=5)
for h in hits:
    print(h.score, h.t_start, h.modality, h.event_id)

# query-by-example: find the audio that matches an image window
dog_img = mem.embedder.embed_image("cam0/dog_01.png")
audio = mem.recall_like(dog_img, modality_in="image", return_modality="audio", k=5)
```

## Wiring in the real embedder

```python
from fusion_embedding.unified import UnifiedEmbedder
from robomem import RobotMemory

embedder = UnifiedEmbedder.from_pretrained(
    "EximiusLabs/fusion-embedding-2-2b-preview",
    device="cuda",
    revision="v0.3-preview",
)
mem = RobotMemory.open("./session.lancedb", embedder=embedder)
mem.index("session.jsonl")
hits = mem.recall("someone handed me a red mug", k=10, after=120.0, before=180.0)
```

Install the real embedder with the `model` extra (`pip install robomem[model]`). The base
package needs only LanceDB, numpy, torch, and Pillow.

## Session manifest

A session is a JSONL file, a JSON array, or an in-memory list of events:

```json
{"t": 12.5, "modality": "image", "path_or_data": "cam0/frame_00012.png", "source": "cam0"}
{"t": 12.5, "modality": "audio", "path_or_data": "mic0/clip_012.wav", "source": "mic0", "duration": 1.0}
{"t": 13.0, "modality": "text",  "path_or_data": "operator said stop", "source": "log"}
```

Required per event: `t` (or `t_start`), `modality`, and `path_or_data`. Optional: `id`,
`source`, `duration` (or `t_end`), `meta`, `thumb`. `modality` is one of
`text | image | video | audio | thermal | motion | geometry`. For image / video / thermal /
geometry / audio, `path_or_data` is normally a path that the embedder loads itself; audio and
motion may also be passed as `{"data": [...], "sr": 16000}`.

## CLI

```
robomem index <manifest> --db <path> [--dedup-tau 0.98] [--fake]
robomem query "<text>" --db <path> [--modality image] [-k 10] [--after 10 --before 30] [--fake]
robomem show <event_id> --db <path> [--fake]
```

`--fake` uses the CPU stand-in embedder (for demos, or to query a store built with the fake).
Without it, the CLI loads `UnifiedEmbedder.from_pretrained(--model, device=--device)`.

## Data model (Tier-0 `events` table)

| column | type | note |
| --- | --- | --- |
| `event_id` | string | unique |
| `t_start`, `t_end` | float64 | window time bounds (seconds) |
| `modality` | string | text / image / video / audio / thermal / motion / geometry |
| `source` | string | stream key (camera / mic / channel) |
| `vector` | list<float32>[2048] | full-width, L2-normalized unified embedding |
| `thumb`, `meta` | string | preview path, JSON metadata |
| `episode_id`, `salience` | string, float32 | **Phase-E hooks**, reserved and unpopulated |

## Retrieval

`recall` embeds the text query, applies a LanceDB SQL prefilter on time / modality, scores an
exact cosine scan over the candidates (session scale, no ANN index), optionally applies
per-modality mean-centering to correct the cross-modal gap, merges temporally-adjacent
same-source hits into segments, and ranks. `recall_like` is the same path starting from a
supplied vector instead of a text query.

## Deferred to the temporal / episodic phase (Phase E)

- Populating `episode_id` and `salience`.
- Episode segmentation (grouping events into coherent episodes).
- Temporal reasoning ("what happened just before X", "how often did Y occur").

The schema and API leave room for these without a migration.

## Development

```
uv venv --system-site-packages
uv pip install lancedb pyarrow pillow pytest
uv run python -m pytest
```

The suite is GPU-free and uses `FakeEmbedder` plus a small synthetic session.
