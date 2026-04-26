from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.graph import run
from src.streaming import stdout_sink


def main() -> int:
    parser = argparse.ArgumentParser(description="cognitive_conversation_engine")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", "-i", required=True, help="user input string")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    on_chunk = None if args.no_stream else stdout_sink()
    final = run(args.input, config=cfg, on_prose_chunk=on_chunk)
    if args.no_stream:
        sys.stdout.write(final.get("prose", "") + "\n")
    else:
        sys.stdout.write("\n")
    flags = final.get("safety", {}).get("flags", {})
    sys.stderr.write(f"[safety] {flags}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
