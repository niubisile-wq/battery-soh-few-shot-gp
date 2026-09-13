#!/usr/bin/env python3
"""Complete slow tail segments by subdividing only their missing byte ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Chunk:
    url: str
    path: Path
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def fetch(chunk: Chunk, attempts: int = 12) -> int:
    if chunk.path.exists() and chunk.path.stat().st_size == chunk.size:
        return chunk.size

    for attempt in range(1, attempts + 1):
        session = requests.Session()
        session.trust_env = False
        try:
            headers = {"Range": f"bytes={chunk.start}-{chunk.end}"}
            with session.get(
                chunk.url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(
                    f"bytes {chunk.start}-{chunk.end}/"
                ):
                    raise RuntimeError(f"bad Content-Range: {content_range}")
                with chunk.path.open("wb") as destination:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            destination.write(block)
            if chunk.path.stat().st_size == chunk.size:
                return chunk.size
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 20))
        finally:
            session.close()
    raise RuntimeError(f"failed: {chunk.path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--segment-mib", type=int, required=True)
    parser.add_argument("--tail-mib", type=int, default=1)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    segment_size = args.segment_mib * 1024 * 1024
    tail_size = args.tail_mib * 1024 * 1024
    incomplete: list[tuple[Path, int, list[Chunk]]] = []
    all_chunks: list[Chunk] = []

    for job in jobs:
        parts_dir = Path(f"{job['output']}.parts")
        for index, start in enumerate(range(0, int(job["size"]), segment_size)):
            end = min(start + segment_size - 1, int(job["size"]) - 1)
            part = parts_dir / f"part_{index:05d}"
            have = part.stat().st_size if part.exists() else 0
            expected = end - start + 1
            if have == expected:
                continue
            if have > expected:
                raise RuntimeError(f"oversized part: {part}")

            tail_dir = Path(f"{part}.tail")
            tail_dir.mkdir(parents=True, exist_ok=True)
            chunks: list[Chunk] = []
            cursor = start + have
            while cursor <= end:
                chunk_end = min(cursor + tail_size - 1, end)
                chunk = Chunk(
                    url=job["url"],
                    path=tail_dir / f"{cursor:012d}_{chunk_end:012d}",
                    start=cursor,
                    end=chunk_end,
                )
                chunks.append(chunk)
                all_chunks.append(chunk)
                cursor = chunk_end + 1
            incomplete.append((part, expected, chunks))

    print(
        f"Completing {len(incomplete)} tails as {len(all_chunks)} chunks "
        f"with {args.workers} workers",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, chunk) for chunk in all_chunks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index % 10 == 0 or index == len(futures):
                print(f"TAIL-PROGRESS {index}/{len(futures)}", flush=True)

    for part, expected, chunks in incomplete:
        with part.open("ab") as destination:
            for chunk in chunks:
                with chunk.path.open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        if part.stat().st_size != expected:
            raise RuntimeError(f"tail assembly failed: {part}")
        shutil.rmtree(Path(f"{part}.tail"))
        print(f"TAIL-COMPLETE {part}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
