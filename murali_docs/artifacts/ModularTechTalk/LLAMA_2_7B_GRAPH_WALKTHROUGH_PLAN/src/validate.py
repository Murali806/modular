#!/usr/bin/env python3
"""Validate source fidelity, complete functions, and local guide navigation."""

import ast
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

from build import REPO, TASK, cpp_span, python_span
from cells import CELLS, REVISION


class GuideParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.dependencies = []
        self.payload = []
        self.in_payload = False
        self.cell = None
        self.in_line = False
        self.lines = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("script", "img", "link", "iframe"):
            src = attrs.get("src") or attrs.get("href")
            if src:
                self.dependencies.append(src)
        if tag == "script" and attrs.get("id") == "guide-data":
            self.in_payload = True
        if tag == "details" and "code-cell" in attrs.get("class", "").split():
            self.cell = attrs["data-cell"]
            self.lines[self.cell] = []
        if tag == "span" and "line-text" in attrs.get("class", "").split() and self.cell:
            self.in_line = True
            self.lines[self.cell].append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_payload = False
        if tag == "span":
            self.in_line = False
        if tag == "details":
            self.cell = None

    def handle_data(self, text):
        if self.in_payload:
            self.payload.append(text)
        if self.in_line:
            self.lines[self.cell][-1] += text


def main():
    manifest = json.loads((TASK / "source_manifest.json").read_text())
    assert manifest["revision"] == REVISION
    parser = GuideParser()
    parser.feed((TASK / "index.html").read_text())
    payload = json.loads("".join(parser.payload))
    assert len(parser.ids) == len(set(parser.ids)), "Duplicate HTML IDs"
    assert not parser.dependencies, f"Unexpected runtime dependencies: {parser.dependencies}"
    assert len([i for i in parser.ids if re.fullmatch(r"chapter-[1-6]", i)]) == 6
    specs = {s["id"]: s for s in CELLS}
    source_bytes = {}
    for file in manifest["files"]:
        raw = (TASK / file["snapshot"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == file["sha256"], file["path"]
        pinned = subprocess.check_output(["git", "show", f'{REVISION}:{file["path"]}'], cwd=REPO)
        assert raw == pinned == (REPO / file["path"]).read_bytes(), file["path"]
        assert payload["files"][file["path"]]["text"].encode() == raw
        source_bytes[file["path"]] = raw
    payload_cells = {c["id"]: c for c in payload["cells"]}
    markdown = (TASK / "WALKTHROUGH.md").read_text()
    line_count = 0
    for cell in manifest["cells"]:
        text = source_bytes[cell["file"]].decode()
        spec = specs[cell["id"]]
        span = (python_span(text, cell["symbol"]) if cell["language"] == "python"
                else cpp_span(text, spec["needle"], spec.get("prefix", 0)))
        assert span == (cell["start"], cell["end"]), cell["id"]
        code = "".join(text.splitlines(keepends=True)[cell["start"] - 1:cell["end"]])
        assert hashlib.sha256(code.encode()).hexdigest() == cell["sha256"]
        assert code == payload_cells[cell["id"]]["code"]
        assert parser.lines[cell["id"]] == code.splitlines(), f'HTML source differs: {cell["id"]}'
        assert f'```{cell["language"]}\n{code.rstrip(chr(10))}\n```' in markdown, f'Markdown source differs: {cell["id"]}'
        for note in cell["notes"]:
            assert cell["start"] <= note["start"] <= note["end"] <= cell["end"]
        line_count += cell["end"] - cell["start"] + 1

    checked_links = 0
    for href in parser.links:
        if href.startswith("http"):
            continue
        path, _, fragment = href.partition("#")
        if path:
            assert (TASK / path).resolve().is_file(), f"Missing HTML link: {href}"
        elif fragment:
            assert fragment in parser.ids, f"Missing HTML anchor: {href}"
        checked_links += 1
    for doc in TASK.glob("*.md"):
        content = doc.read_text()
        assert sum(s.startswith("```") for s in content.splitlines()) % 2 == 0, doc
        clean = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", content, flags=re.S | re.M)
        for target in re.findall(r"\[[^\]\n]+\]\(([^\s)]+)\)", clean):
            if "://" in target:
                continue
            path, _, fragment = target.partition("#")
            resolved = (doc.parent / path).resolve()
            assert resolved.is_file(), f"Missing Markdown link: {doc.name}: {target}"
            if re.fullmatch(r"L\d+", fragment):
                assert 1 <= int(fragment[1:]) <= len(resolved.read_text().splitlines())
            checked_links += 1
    for file in (TASK / "src").glob("*.py"):
        ast.parse(file.read_text(), filename=str(file))
    report = dict(
        checked_at=datetime.now(timezone.utc).isoformat(), revision=REVISION,
        command="PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py",
        result="PASS", source_files=len(manifest["files"]),
        complete_source_cells=len(manifest["cells"]), displayed_source_lines=line_count,
        local_links_checked=checked_links, external_runtime_dependencies=0,
        evidence="Document/source validation only; no MAX model execution.",
    )
    (TASK / "validation").mkdir(exist_ok=True)
    (TASK / "validation" / "source_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
