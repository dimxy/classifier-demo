from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass
class StreamChannel:
    name: str
    enabled: bool
    visibility: str
    sink: Callable[[str], None]


def stdout_sink(prefix: str = "") -> Callable[[str], None]:
    def _write(chunk: str) -> None:
        sys.stdout.write(prefix + chunk)
        sys.stdout.flush()
    return _write


def make_channels(
    stream_thoughts: bool,
    visibility: str = "all_users",
) -> dict[str, StreamChannel]:
    channels: dict[str, StreamChannel] = {
        "prose_stream": StreamChannel(
            name="prose_stream",
            enabled=True,
            visibility="all_users",
            sink=stdout_sink(),
        ),
    }
    if stream_thoughts and visibility == "premium_only":
        channels["thought_stream"] = StreamChannel(
            name="thought_stream",
            enabled=True,
            visibility="premium_only",
            sink=stdout_sink("[think] "),
        )
    return channels
