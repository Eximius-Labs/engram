"""robomem — the robot-memory layer for Eximius Labs.

Ingest a recorded multimodal session, index it on the unified embedding space, and answer
natural-language recall queries over it. robomem holds no model of its own: it takes an
embedder by dependency injection (``fusion_embedding.unified.UnifiedEmbedder`` in production,
``robomem.fakes.FakeEmbedder`` in tests).
"""

from .embedder import Embedder
from .fakes import FakeEmbedder
from .memory import RobotMemory
from .schema import VECTOR_DIM, Moment

__all__ = ["RobotMemory", "Embedder", "FakeEmbedder", "Moment", "VECTOR_DIM"]
__version__ = "0.1.0"
