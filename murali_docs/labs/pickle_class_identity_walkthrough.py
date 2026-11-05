#!/usr/bin/env python3
"""Show a successful config transfer and a sender-side class identity failure.

CPU-RUN teaching experiment using only locally generated Python code and the
standard library. A fresh isolated subprocess receives ordinary pickle bytes
over a pipe. No Hugging Face code, MAX runtime or model weights are loaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import pickle
import pickletools
import platform
import subprocess
import sys
import tempfile
from typing import Any


MODULE_NAME = "model_types"
MODULE_CODE = """class DemoConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
"""

WORKER_CODE = """
import json
import pickle
import sys

# This directory contains the locally generated model_types.py file.
sys.path.insert(0, sys.argv[1])
cfg = pickle.loads(sys.stdin.buffer.read())
import model_types

print(json.dumps({
    'outcome': 'success',
    'state': vars(cfg),
    'module': type(cfg).__module__,
    'class_name': type(cfg).__qualname__,
    'matches_worker_class': type(cfg) is model_types.DemoConfig,
}))
"""


def _receive_in_fresh_worker(payload: bytes, source: Path) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-I", "-c", WORKER_CODE, str(source)],
        input=payload,
        capture_output=True,
        check=True,
        timeout=20,
    )
    return json.loads(process.stdout)


def run_walkthrough() -> dict[str, Any]:
    """Run import, serialization, reload and fresh-worker reconstruction steps."""
    if MODULE_NAME in sys.modules:
        raise RuntimeError("Run this standalone lab in a fresh Python interpreter")

    with tempfile.TemporaryDirectory(prefix="pickle-class-identity-") as directory:
        source = Path(directory)
        (source / f"{MODULE_NAME}.py").write_text(MODULE_CODE)
        sys.path.insert(0, directory)
        try:
            model_types = importlib.import_module(MODULE_NAME)
            cfg = model_types.DemoConfig(hidden_size=4096)
            class_a = type(cfg)
            ordinary_import = importlib.import_module(MODULE_NAME)

            payload = pickle.dumps(cfg, protocol=4)
            restored = pickle.loads(payload)
            normal = {
                "sender_class_lookup_matches": type(cfg) is model_types.DemoConfig,
                "repeated_import_preserves_class": ordinary_import.DemoConfig is class_a,
                "pickle_protocol": 4,
                "pickle_strings": [
                    value for _, value, _ in pickletools.genops(payload)
                    if isinstance(value, str)
                ],
                "local_roundtrip": {
                    "same_instance": restored is cfg,
                    "same_class": type(restored) is class_a,
                    "same_state": vars(restored) == vars(cfg),
                },
                "worker": _receive_in_fresh_worker(payload, source),
            }

            # Re-executing the same file creates a new DemoConfig class object.
            # The old cfg instance continues to refer to class_a.
            importlib.reload(model_types)
            class_b = model_types.DemoConfig
            reloaded = {
                "source_code_changed": False,
                "same_module_name": class_a.__module__ == class_b.__module__,
                "same_class_name": class_a.__qualname__ == class_b.__qualname__,
                "same_class_object": class_a is class_b,
                "instance_still_uses_class_a": type(cfg) is class_a,
                "sender_class_lookup_matches": type(cfg) is model_types.DemoConfig,
                "state": vars(cfg),
            }
            try:
                pickle.dumps(cfg, protocol=4)
            except pickle.PicklingError as error:
                reloaded["outcome"] = type(error).__name__
                reloaded["message"] = str(error)
                reloaded["phase"] = "parent serialization; no completed pickle returned"
            else:
                raise RuntimeError("The old instance unexpectedly serialized after reload")

            # A new instance uses the currently published class B.
            new_cfg = model_types.DemoConfig(hidden_size=cfg.hidden_size)
            recreated = {
                "sender_class_lookup_matches": type(new_cfg) is model_types.DemoConfig,
                "worker": _receive_in_fresh_worker(pickle.dumps(new_cfg, protocol=4), source),
            }
        finally:
            sys.path.remove(directory)
            sys.modules.pop(MODULE_NAME, None)

    assert normal["sender_class_lookup_matches"]
    assert normal["repeated_import_preserves_class"]
    assert normal["local_roundtrip"] == {
        "same_instance": False, "same_class": True, "same_state": True
    }
    assert normal["worker"]["matches_worker_class"]
    assert normal["worker"]["state"] == {"hidden_size": 4096}
    assert not reloaded["same_class_object"]
    assert reloaded["same_class_name"] and reloaded["same_module_name"]
    assert reloaded["instance_still_uses_class_a"]
    assert not reloaded["sender_class_lookup_matches"]
    assert recreated["sender_class_lookup_matches"]
    assert recreated["worker"]["matches_worker_class"]
    assert recreated["worker"]["state"] == {"hidden_size": 4096}

    return {
        "evidence": "CPU-RUN",
        "python": platform.python_version(),
        "scope": "Synthetic local class; ordinary pickle bytes through a pipe to fresh isolated interpreters; no HF/MAX model execution",
        "normal_transfer": normal,
        "parent_after_reload": reloaded,
        "new_instance_after_reload": recreated,
    }


def main() -> None:
    """Print the observed results and optionally save the JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(run_walkthrough(), indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
