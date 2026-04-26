from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("cce")


@contextmanager
def time_node(name: str, payload: Dict[str, Any] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("node.start name=%s input=%s", name, _truncate(payload))
    try:
        yield
    finally:
        dur_ms = (time.perf_counter() - start) * 1000
        logger.info("node.end name=%s latency_ms=%.1f", name, dur_ms)


def log_node_output(name: str, output: Dict[str, Any]) -> None:
    logger.info("node.output name=%s output=%s", name, _truncate(output))


def _truncate(obj: Any, n: int = 200) -> str:
    s = str(obj)
    return s if len(s) <= n else s[:n] + "..."
