#!/usr/bin/env python3
"""Strict post-download deployment check for paper 2."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "数据集/BatteryLife_v12_processed/range_manifest_selected_v12.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    failures: list[str] = []
    for item in json.loads(MANIFEST.read_text(encoding="utf-8")):
        path = Path(item["output"])
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        if path.stat().st_size != int(item["size"]):
            failures.append(f"size: {path} ({path.stat().st_size} != {item['size']})")
            continue
        if item.get("md5"):
            md5 = hashlib.md5()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    md5.update(block)
            digest = md5.hexdigest()
            if digest != item["md5"]:
                failures.append(f"md5: {path} ({digest} != {item['md5']})")
        if path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                if bad is not None:
                    failures.append(f"zip member: {path}: {bad}")
            except Exception as exc:
                failures.append(f"zip: {path}: {exc}")

    repos = {
        "BatteryML": "2861ae3b8c79938c7fc8e6fe9986b799ca71c7dd",
        "BatteryLife": "3b17bb53202e10711deae0448c91f84144bb3dd9",
        "PINN4SOH": "cc3cc5053caf38f04e0665f7f88cb109144d035e",
        "SOHbenchmark": "62230dd9a857f5c05ac641227b52ec2ffbeb9e15",
    }
    for name, expected in repos.items():
        repo = ROOT / "基线模型/参考实现" / name
        try:
            actual = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != expected:
                failures.append(f"commit: {name} ({actual} != {expected})")
        except Exception as exc:
            failures.append(f"repo: {name}: {exc}")

    if failures:
        print("DEPLOYMENT_INCOMPLETE")
        print("\n".join(failures))
        return 1
    print("DEPLOYMENT_VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
