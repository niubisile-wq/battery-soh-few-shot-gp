#!/usr/bin/env python3
"""Resumable parallel range downloader for the large official MATR files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Part:
    dataset_name: str
    url: str
    path: Path
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def download_part(
    part: Part,
    attempts: int = 12,
    use_env_proxy: bool = False,
) -> tuple[str, int]:
    part.path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        have = part.path.stat().st_size if part.path.exists() else 0
        if have == part.size:
            return part.dataset_name, part.size
        if have > part.size:
            part.path.unlink()
            have = 0

        requested_start = part.start + have
        headers = {"Range": f"bytes={requested_start}-{part.end}"}
        session = requests.Session()
        session.trust_env = use_env_proxy

        try:
            with session.get(
                part.url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"expected HTTP 206, received {response.status_code}"
                    )

                expected_range = f"bytes {requested_start}-{part.end}/"
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(expected_range):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")

                with part.path.open("ab") as destination:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            destination.write(block)

            if part.path.stat().st_size == part.size:
                return part.dataset_name, part.size
        except Exception as exc:
            if attempt == attempts:
                raise RuntimeError(f"{part.path.name}: {exc}") from exc
            time.sleep(min(2**attempt, 30))
        finally:
            session.close()

    raise RuntimeError(f"failed to download {part.path}")


def prepare_parts(job: dict, segment_size: int) -> tuple[Path, int, list[Part]]:
    output = Path(job["output"])
    expected_size = int(job["size"])
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and output.stat().st_size == expected_size:
        return output, expected_size, []

    parts_dir = Path(f"{output}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    first_part = parts_dir / "part_00000"
    if output.exists():
        current_size = output.stat().st_size
        if current_size > segment_size:
            raise RuntimeError(
                f"cannot reuse {output}: partial file is larger than one segment"
            )
        if first_part.exists():
            raise RuntimeError(
                f"both partial output and first segment exist for {output}"
            )
        os.replace(output, first_part)

    parts: list[Part] = []
    index = 0
    start = 0
    while start < expected_size:
        end = min(start + segment_size - 1, expected_size - 1)
        parts.append(
            Part(
                dataset_name=job["name"],
                url=job["url"],
                path=parts_dir / f"part_{index:05d}",
                start=start,
                end=end,
            )
        )
        index += 1
        start = end + 1

    return output, expected_size, parts


def assemble(output: Path, expected_size: int, parts: list[Part]) -> str:
    if not parts:
        digest = hashlib.sha256()
        with output.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    temporary = Path(f"{output}.assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as destination:
        for part in parts:
            if part.path.stat().st_size != part.size:
                raise RuntimeError(f"incomplete segment: {part.path}")
            with part.path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(block)
                    digest.update(block)
        destination.flush()
        os.fsync(destination.fileno())

    if temporary.stat().st_size != expected_size:
        raise RuntimeError(f"assembled size mismatch for {output}")

    os.replace(temporary, output)
    shutil.rmtree(Path(f"{output}.parts"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--segment-mib", type=int, default=64)
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="honor the current environment proxy settings",
    )
    args = parser.parse_args()

    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
    segment_size = args.segment_mib * 1024 * 1024

    prepared = [prepare_parts(job, segment_size) for job in jobs]
    tasks = [part for _, _, parts in prepared for part in parts]
    total = sum(part.size for part in tasks)
    initial_sizes = {
        part: min(part.path.stat().st_size, part.size) if part.path.exists() else 0
        for part in tasks
    }
    already = sum(initial_sizes.values())
    completed = already
    started = time.monotonic()

    print(
        f"Downloading {len(tasks)} segments with {args.workers} workers; "
        f"reusing {already / 1e6:.1f} MB",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_part = {
            pool.submit(download_part, part, 12, args.use_env_proxy): part
            for part in tasks
        }
        last_report = 0.0
        for future in concurrent.futures.as_completed(future_to_part):
            part = future_to_part[future]
            try:
                _, part_size = future.result()
            except Exception as exc:
                print(f"ERROR: {part.dataset_name}/{part.path.name}: {exc}", file=sys.stderr)
                return 1

            completed += part_size - initial_sizes[part]
            now = time.monotonic()
            if now - last_report >= 15 or completed >= total:
                elapsed = max(now - started, 0.001)
                downloaded_now = max(completed - already, 0)
                speed = downloaded_now / elapsed / 1e6
                progress = min(completed / max(total, 1) * 100, 100)
                print(
                    f"PROGRESS {progress:5.1f}%  "
                    f"{completed / 1e9:.2f}/{total / 1e9:.2f} GB  "
                    f"{speed:.2f} MB/s",
                    flush=True,
                )
                last_report = now

    for output, expected_size, parts in prepared:
        sha256 = assemble(output, expected_size, parts)
        print(f"COMPLETE {output} sha256={sha256}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
