#!/usr/bin/env python3
"""Refresh small, pinned HF metadata; the HTML build itself stays offline."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

TASK = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-2-7b-hf"
MODEL_REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
MIRROR_ID = "NousResearch/Llama-2-7b-hf"
MIRROR_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"


def get_json(url):
    request = Request(url, headers={"User-Agent": "MAX-source-walkthrough/1.0"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
        return json.loads(raw), response.status, hashlib.sha256(raw).hexdigest()


def main():
    out = TASK / "evidence"
    out.mkdir(exist_ok=True)
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{MODEL_REVISION}?blobs=true"
    info, status, digest = get_json(api_url)
    assert info["sha"] == MODEL_REVISION
    selected = [item for item in info["siblings"]
                if item["rfilename"] in ("model-00001-of-00002.safetensors",
                                         "model-00002-of-00002.safetensors")]
    assert len(selected) == 2
    official_config_url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/config.json"
    try:
        official_config, config_status, _ = get_json(official_config_url)
    except HTTPError as error:
        official_config, config_status = None, error.code
    mirror_url = f"https://huggingface.co/{MIRROR_ID}/resolve/{MIRROR_REVISION}/config.json"
    mirror_config, mirror_status, mirror_digest = get_json(mirror_url)
    (out / "llama2_7b_public_mirror_config.json").write_text(
        json.dumps(mirror_config, indent=2) + "\n")
    evidence = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "method": "Unauthenticated HTTPS metadata requests; no model weights downloaded.",
        "official": {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "api_url": api_url, "http_status": status, "response_sha256": digest,
            "selected_weight_files": selected,
            "total_selected_weight_bytes": sum(item["size"] for item in selected),
            "safetensors_metadata": info.get("safetensors"),
            "config_url": official_config_url, "config_http_status": config_status,
            "config": official_config,
        },
        "public_mirror": {
            "model_id": MIRROR_ID, "revision": MIRROR_REVISION,
            "url": mirror_url, "http_status": mirror_status,
            "response_sha256": mirror_digest,
            "snapshot": "evidence/llama2_7b_public_mirror_config.json",
            "snapshot_sha256": hashlib.sha256(
                (out / "llama2_7b_public_mirror_config.json").read_bytes()).hexdigest(),
        },
        "example_policy": {
            "free_device_memory_gib": 24, "device_memory_utilization": 0.9,
            "cache_bytes_per_element": 2, "kv_cache_page_size": 128,
            "max_batch_size": 4, "max_length": 4096,
            "kind": "Hypothetical single-device arithmetic; no serving or GPU measurement.",
        },
    }
    (out / "llama2_7b_memory_planning_provenance.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"result": "Saved pinned metadata", "weight_bytes": evidence["official"]["total_selected_weight_bytes"],
                      "official_config_http_status": config_status}, indent=2))


if __name__ == "__main__":
    main()
