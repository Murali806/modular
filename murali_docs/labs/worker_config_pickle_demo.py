#!/usr/bin/env python3
"""Demonstrate dynamic-class pickle boundaries using only the Python stdlib.

CPU-RUN teaching experiment. All imported demo code is created by this script
inside a temporary directory. No Hub code, model weights or MAX runtime are
loaded. A fresh isolated Python interpreter models the import boundary of spawn.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
from typing import Any


MODULE_NAME = "worker_config_demo"
MODULE_CODE = """class DemoConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
"""

CHILD_CODE = """
import importlib.util
import json
from pathlib import Path
import pickle
import sys

try:
    received = pickle.load(sys.stdin.buffer)
    if isinstance(received, dict):
        source = Path(received['model_dir'])
        name = received['module_name']
        spec = importlib.util.spec_from_file_location(name, source / 'configuration_demo.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        config = getattr(module, received['class_name'])(
            **json.loads((source / 'config.json').read_text())
        )
    else:
        config = received
    print(json.dumps({'outcome': 'success', 'hidden_size': config.hidden_size}))
except Exception as error:
    print(json.dumps({'outcome': type(error).__name__, 'message': str(error)}))
    sys.exit(1)
"""


def _load_demo_class(source: Path) -> type:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, source / "configuration_demo.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the local demonstration module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module.DemoConfig


def _fresh_interpreter(payload: bytes) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-I", "-c", CHILD_CODE],
        input=payload,
        capture_output=True,
        timeout=20,
        check=False,
    )
    result = json.loads(process.stdout)
    result["exit_code"] = process.returncode
    return result


def main() -> None:
    """Run four scenarios and optionally save their observed outcomes as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="max-config-pickle-") as directory:
        source = Path(directory)
        (source / "configuration_demo.py").write_text(MODULE_CODE)
        (source / "config.json").write_text('{"hidden_size": 4096}\n')
        original_class = _load_demo_class(source)
        config = original_class(hidden_size=4096)
        payload = pickle.dumps(config)
        roundtrip = pickle.loads(payload)
        same_process = {
            "outcome": "success",
            "hidden_size": roundtrip.hidden_size,
            "same_class": type(roundtrip) is original_class,
        }

        missing_import = _fresh_interpreter(payload)

        replacement_class = _load_demo_class(source)
        try:
            pickle.dumps(config)
            changed_identity = {"outcome": "unexpected_success"}
        except pickle.PicklingError as error:
            changed_identity = {
                "outcome": type(error).__name__,
                "message": str(error),
                "same_class_name": original_class.__name__ == replacement_class.__name__,
                "same_class_object": original_class is replacement_class,
            }

        source_state = {
            "model_dir": str(source),
            "module_name": MODULE_NAME,
            "class_name": "DemoConfig",
        }
        rebuilt = _fresh_interpreter(pickle.dumps(source_state))

    sys.modules.pop(MODULE_NAME, None)
    if not (
        same_process["hidden_size"] == 4096
        and same_process["same_class"]
        and missing_import["outcome"] == "ModuleNotFoundError"
        and missing_import["exit_code"] == 1
        and changed_identity["outcome"] == "PicklingError"
        and rebuilt["outcome"] == "success"
        and rebuilt["hidden_size"] == 4096
        and rebuilt["exit_code"] == 0
    ):
        raise RuntimeError("The observed behavior differs from the documented experiment")

    report = {
        "evidence": "CPU-RUN",
        "python": platform.python_version(),
        "scope": "Synthetic dynamic class; fresh isolated interpreters; no HF or MAX model execution",
        "same_process_roundtrip": same_process,
        "worker_missing_import": missing_import,
        "parent_class_identity_changed": changed_identity,
        "worker_rebuild_from_source": rebuilt,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
