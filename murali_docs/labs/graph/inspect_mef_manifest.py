#!/usr/bin/env python3
"""Summarize exported MEFs without loading or executing them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _graph_summary(directory: Path, graph: dict[str, Any]) -> dict[str, Any]:
    artifact = directory / graph["key"]
    return {
        "name": graph["name"],
        "file": graph["key"],
        "bytes": artifact.stat().st_size,
        "sha256": _sha256(artifact),
        "manifest_fingerprint": graph["fingerprint"],
        "inputs": graph["signature"]["inputs"],
        "outputs": graph["signature"]["outputs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    graphs = [
        _graph_summary(args.directory, graph) for graph in manifest["graphs"]
    ]
    print(
        json.dumps(
            {
                "evidence": "COMPILE-ONLY",
                "directory": str(args.directory),
                "graph_count": len(graphs),
                "total_mef_bytes": sum(graph["bytes"] for graph in graphs),
                "graphs": graphs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
