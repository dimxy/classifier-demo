from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Protocol

from ..observability import logger


class _Embedder(Protocol):
    vector_size: int

    def embed(self, text: str) -> Optional[List[float]]: ...


class ChatStore:
    """Qdrant-local-mode wrapper for cross-run conversational memory.

    All operations swallow errors and log; callers should never see exceptions
    raised from this class so the pipeline keeps running.
    """

    def __init__(
        self,
        path: str,
        collection: str,
        embedder: _Embedder,
        vector_size: Optional[int] = None,
    ) -> None:
        self.path = path
        self.collection = collection
        self.embedder = embedder
        self.vector_size = vector_size or getattr(embedder, "vector_size", 768)
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qm

            self._client = QdrantClient(path=self.path)
            existing = {c.name for c in self._client.get_collections().collections}
            if self.collection not in existing:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(
                        size=self.vector_size, distance=qm.Distance.COSINE
                    ),
                )
        except Exception as exc:
            logger.error("ChatStore init failed: %s", exc)
            self._client = None

    def recall(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._client is None:
            return []
        vec = self.embedder.embed(query_text)
        if vec is None:
            return []
        try:
            resp = self._client.query_points(
                collection_name=self.collection, query=vec, limit=k, with_payload=True
            )
            hits = resp.points
        except Exception as exc:
            logger.error("ChatStore.recall failed: %s", exc)
            return []
        out: List[Dict[str, Any]] = []
        for h in hits:
            payload = dict(h.payload or {})
            payload["score"] = float(h.score)
            out.append(payload)
        return out

    def write(self, user_input: str, prose: str) -> None:
        if self._client is None:
            return
        vec = self.embedder.embed(user_input)
        if vec is None:
            return
        try:
            from qdrant_client.http import models as qm

            point = qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "user_input": user_input,
                    "prose": prose,
                    "ts": time.time(),
                },
            )
            self._client.upsert(collection_name=self.collection, points=[point])
        except Exception as exc:
            logger.error("ChatStore.write failed: %s", exc)
