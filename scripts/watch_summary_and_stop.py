"""Stop a resumable runner after its summary reaches a processed threshold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def read_processed(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("processed") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def process_group(pid: int) -> int:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    last = -1
    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            print(f"process {args.pid} already exited", file=open(args.log, "a", encoding="utf-8"))
            return
        processed = read_processed(args.summary)
        if processed != last:
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(f"processed={processed} threshold={args.threshold}\n")
            last = processed
        if processed >= args.threshold:
            pgid = process_group(args.pid)
            target = pgid if pgid > 0 else args.pid
            with args.log.open("a", encoding="utf-8") as stream:
                stream.write(f"stopping pgid={target} at processed={processed}\n")
            try:
                os.killpg(target, signal.SIGTERM)
            except ProcessLookupError:
                return
            time.sleep(3)
            try:
                os.killpg(target, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
