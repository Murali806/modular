"""Save small public-reference extracts for the offline OS/GPU systems page.

Run deliberately when refreshing research. The normal builder never fetches
network resources. These extracts document concepts, not MAX runtime timings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.request

TASK = Path(__file__).resolve().parent.parent
SOURCES = [
    ("nginx", "NGINX: event-driven server architecture",
     "https://blog.nginx.org/blog/inside-nginx-how-we-designed-for-performance-scale",
     r"nonblocking|non.blocking|single.threaded|state machine|worker process handles"),
    ("balancing", "Sam Who: interactive Load Balancing",
     "https://samwho.dev/load-balancing/",
     r"round robin|request queue|Queues help|queue is full|least connections|two random|power of two"),
    ("batching", "Anyscale: continuous batching for LLM inference",
     "https://www.anyscale.com/blog/continuous-batching-llm-inference",
     r"Completing four|Completing seven|iteration.level scheduling|new sequence can be inserted|buffer allocation can happen"),
    ("ray", "Ray Serve: proxies, replica queues and failure recovery",
     "https://docs.ray.io/en/latest/serve/architecture.html",
     r"HTTP Proxy:|Replicas:|request is left in the queue|Actors fail|Autoscal"),
    ("libuv", "libuv: design of the I/O event loop",
     "https://docs.libuv.org/en/v1.x/design.html",
     r"tied to a single thread|epoll on Linux|thread pool|block waiting for I/O"),
    ("epoll", "Linux epoll: interest and ready lists",
     "https://man7.org/linux/man-pages/man7/epoll.7.html",
     r"interest list|ready list|epoll_wait|nonblocking|EAGAIN"),
    ("napi", "Linux NAPI: network-driver event processing",
     "https://docs.kernel.org/networking/napi.html",
     r"NAPI is the event|software interrupt context|polling occurs|IRQ|interrupt mitigation"),
    ("dma", "Linux DMA API: CPU, physical and device addresses",
     "https://docs.kernel.org/core-api/dma-api-howto.html",
     r"three different addresses|bus address|IOMMU|DMA engine|DMA_TO_DEVICE"),
    ("memory", "Linux memory-management concepts",
     "https://docs.kernel.org/admin-guide/mm/concepts.html",
     r"4.?K|virtual address|page cache|physical memory|huge pages"),
    ("cgroup", "Linux cgroup v2: CPU and RAM limits",
     "https://docs.kernel.org/admin-guide/cgroup-v2.html",
     r"^cpu.max\b|^memory.max\b|CPU bandwidth|memory usage reaches the limit"),
    ("fdlimit", "Linux getrlimit: open-file-descriptor limits",
     "https://man7.org/linux/man-pages/man2/getrlimit.2.html",
     r"RLIMIT_NOFILE|EMFILE|open file descriptors"),
    ("spawn", "Python multiprocessing: spawn and process lifecycle",
     "https://docs.python.org/3/library/multiprocessing.html",
     r"fresh Python interpreter|unnecessary file descriptors|wait on the sentinel|process exited|sentinel.*file descriptor"),
    ("uvicorn", "Uvicorn: flow control and server limits",
     "https://uvicorn.dev/server-behavior/",
     r"high water mark|Flow Control|flow control|--limit-concurrency and --backlog|application.level gate"),
    ("zmq", "ZeroMQ ipc transport: local Unix-domain sockets",
     "https://libzmq.readthedocs.io/en/latest/zmq_ipc.html",
     r"passes messages between local|UNIX domain sockets|pathname.*unique|abstract namespace"),
    ("cuda_async", "NVIDIA CUDA: asynchronous execution, streams and events",
     "https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html",
     r"dispatching function|sequence of operations|order of execution|Streams and CUDA Events|synchronization|page.locked|pinned"),
    ("cuda_driver", "NVIDIA CUDA: driver API and context address spaces",
     "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html",
     r"lower level CUDA driver API|dynamic library|context is analogous|distinct address space|cuInit"),
    ("nvidia_kernel", "NVIDIA Linux GPU kernel modules and user-space components",
     "https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/main/README.md",
     r"source release|must be used with GSP firmware|user.space NVIDIA GPU driver"),
]


class VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip += 1
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "pre") and not self.skip:
            self.parts.append("\n\n")

    def handle_endtag(self, tag) -> None:
        if tag in ("script", "style", "nav", "header", "footer") and self.skip:
            self.skip -= 1
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "pre") and not self.skip:
            self.parts.append("\n\n")

    def handle_data(self, value) -> None:
        if not self.skip:
            self.parts.append(value)


def brief_excerpts(paragraphs, budget=180):
    """Keep brief attributed extracts; full articles remain at their links."""
    result = []
    for paragraph in paragraphs:
        words = list(re.finditer(r"\S+", paragraph))
        if budget < 15:
            break
        if len(words) > budget:
            paragraph = paragraph[:words[budget - 1].end()]
            ends = list(re.finditer(r"[.!?](?:\s|$)", paragraph))
            if ends and ends[-1].start() > 60:
                paragraph = paragraph[:ends[-1].start() + 1]
        result.append(paragraph)
        budget -= len(paragraph.split())
    return result


def fetch(spec):
    key, title, url, pattern = spec
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (documentation review)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(4_000_000)
        final_url = response.url
    decoded = raw.decode("utf-8", "replace")
    if url.endswith(".md"):
        visible = decoded
    else:
        parser = VisibleText()
        parser.feed(decoded)
        visible = "".join(parser.parts)
    paragraphs = [re.sub(r"\s+", " ", block).strip() for block in visible.split("\n\n")]
    selected = []
    for index, paragraph in enumerate(paragraphs):
        match = re.search(pattern, paragraph, re.I)
        if not match:
            continue
        if len(paragraph) < 70:
            # Kernel documentation often separates a field name from its
            # description. Keep neighboring prose so cpu.max is explained.
            neighbors = [p for p in paragraphs[index + 1:index + 7] if len(p) > 40][:2]
            paragraph = " ".join([paragraph, *neighbors])
        if len(paragraph) < 70:
            continue
        if len(paragraph) > 2200:
            # Man-page pre blocks occasionally span a section. Retain a
            # documented extract around the matching concept, not the menu.
            start = max(0, paragraph.rfind(". ", 0, match.start()) + 2)
            end = paragraph.find(". ", match.end() + 500)
            paragraph = paragraph[start:end + 1 if end >= 0 else start + 2000]
        if paragraph not in selected:
            selected.append(paragraph)
        if len(selected) == 7:
            break
    if not selected:
        raise ValueError(f"No research paragraphs matched {key}: {final_url}")
    return dict(id=key, title=title, url=final_url, paragraphs=brief_excerpts(selected),
                source_sha256=hashlib.sha256(raw).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Explicitly fetch the public references")
    args = parser.parse_args()
    if not args.refresh:
        parser.error("Use --refresh to deliberately fetch references; normal builds are offline.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES))
    fetched_at = datetime.now(timezone.utc).isoformat()
    destination = TASK / "evidence" / "os_systems"
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        snapshot = destination / (record["id"] + ".txt")
        text = record["title"] + "\n" + record["url"] + "\nRetrieved: " + fetched_at + "\n\n"
        text += "\n\n".join(record["paragraphs"]) + "\n"
        snapshot.write_text(text)
        record.update(snapshot=str(snapshot.relative_to(TASK)),
                      snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        print(json.dumps({"id": record["id"], "url": record["url"],
                          "paragraphs": len(record["paragraphs"]),
                          "sample": record["paragraphs"][0][:500]}))
    result = dict(retrieved_at=fetched_at, evidence="Public documentation and visual-design references",
                  records=records)
    (TASK / "evidence" / "os_systems_research.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
