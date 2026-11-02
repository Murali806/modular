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
from call_stack import READER_NAME, TEXT_NAME
from cells import CELLS, REVISION
from memory_planning import PAGE_NAME


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


class StackParser(GuideParser):
    def __init__(self):
        super().__init__()
        self.trees = []
        self.in_tree = False

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "stack-data":
            self.in_payload = True
        if tag == "pre" and "text-tree" in attrs.get("class", "").split():
            self.in_tree = True
            self.trees.append("")

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag == "pre":
            self.in_tree = False

    def handle_data(self, text):
        super().handle_data(text)
        if self.in_tree:
            self.trees[-1] += text


class MemoryParser(GuideParser):
    def __init__(self):
        super().__init__()
        self.disclosure_depth = 0
        self.max_disclosure_depth = 0
        self.qna_cells = 0

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == "memory-data":
            self.in_payload = True
        if tag == "details":
            self.disclosure_depth += 1
            self.max_disclosure_depth = max(self.max_disclosure_depth, self.disclosure_depth)
            if "qna" in attrs.get("class", "").split():
                self.qna_cells += 1

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag == "details":
            self.disclosure_depth -= 1


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

    stack_parser = StackParser()
    stack_parser.feed((TASK / READER_NAME).read_text())
    stack_payload = json.loads("".join(stack_parser.payload))
    stack_manifest = json.loads((TASK / "call_stack_manifest.json").read_text())
    assert stack_manifest["revision"] == REVISION
    assert stack_payload["references"] == stack_manifest["references"]
    assert stack_payload["sections"] == stack_manifest["sections"]
    assert stack_payload["text"] == (TASK / TEXT_NAME).read_text()
    # HTML displays a local document's clickable label; its target path is
    # already carried by href. The plain-text companion also prints that path.
    expected_visible_trees = [
        re.sub(r" \(link_htmls/[^)\n]+\)", "", section["text"])
        for section in stack_manifest["sections"]
    ]
    assert stack_parser.trees == expected_visible_trees
    assert len(stack_parser.trees) == 1
    assert len(stack_parser.ids) == len(set(stack_parser.ids))
    assert not stack_parser.dependencies
    for ref in stack_manifest["references"].values():
        source = source_bytes[ref["file"]].decode()
        assert source.splitlines()[ref["line"] - 1] == ref["text"]
        assert ref["start"] <= ref["line"] <= ref["end"]
        if ref.get("symbol"):
            assert python_span(source, ref["symbol"]) == (ref["start"], ref["end"])
        assert stack_payload["files"][ref["file"]]["text"] == source
    assert "{{" not in stack_payload["text"]

    memory_page = TASK / "link_htmls" / PAGE_NAME
    memory_parser = MemoryParser()
    memory_parser.feed(memory_page.read_text())
    memory_payload = json.loads("".join(memory_parser.payload))
    memory_manifest = json.loads((TASK / "memory_planning_manifest.json").read_text())
    assert memory_manifest["revision"] == memory_payload["revision"] == REVISION
    assert len(memory_parser.ids) == len(set(memory_parser.ids)), "Duplicate memory-page IDs"
    assert not memory_parser.dependencies, "Memory page must work offline"
    assert memory_parser.disclosure_depth == 0
    assert memory_parser.max_disclosure_depth >= 3, "Expected nested expandable explanations/code"
    assert memory_parser.qna_cells == 9, "Expected root QnA and eight child sections"
    assert memory_manifest["complete_code_cells"] == memory_payload["cells"] == sorted(memory_parser.lines)
    memory_line_count = 0
    for key, ref in memory_payload["references"].items():
        recorded = memory_manifest["references"][key]
        assert recorded == {k: v for k, v in ref.items() if k != "code"}
        source = source_bytes[ref["file"]].decode()
        assert python_span(source, ref["symbol"]) == (ref["start"], ref["end"]), key
        code = "".join(source.splitlines(keepends=True)[ref["start"] - 1:ref["end"]])
        assert ref["code"] == code, key
        assert hashlib.sha256(code.encode()).hexdigest() == ref["sha256"], key
        if ref.get("needle"):
            assert code.count(ref["needle"]) == 1, key
            assert ref["line"] == ref["start"] + code[:code.index(ref["needle"])].count("\n"), key
        assert memory_payload["files"][ref["file"]]["text"] == source
        if key in memory_parser.lines:
            assert memory_parser.lines[key] == code.splitlines(), f"Memory HTML source differs: {key}"
            memory_line_count += ref["end"] - ref["start"] + 1
    for evidence in memory_manifest["evidence_files"]:
        assert hashlib.sha256((TASK / evidence["path"]).read_bytes()).hexdigest() == evidence["sha256"]
    provenance = memory_payload["evidence"]
    assert provenance == json.loads((TASK / "evidence/llama2_7b_memory_planning_provenance.json").read_text())
    assert provenance["official"]["total_selected_weight_bytes"] == sum(
        item["size"] for item in provenance["official"]["selected_weight_files"])
    assert provenance["official"]["total_selected_weight_bytes"] == memory_payload["example"]["weightBytes"]
    assert hashlib.sha256((TASK / provenance["public_mirror"]["snapshot"]).read_bytes()).hexdigest() == provenance["public_mirror"]["snapshot_sha256"]

    checked_links = 0
    page_parsers = {TASK / "index.html": parser, TASK / READER_NAME: stack_parser, memory_page: memory_parser}
    for page, parsed in list(page_parsers.items()):
        for href in parsed.links:
            if href.startswith("http"):
                continue
            path, _, fragment = href.partition("#")
            target = (page.parent / path).resolve() if path else page.resolve()
            assert target.is_file(), f"Missing HTML link: {page.name}: {href}"
            if fragment and target.suffix == ".html":
                if target not in page_parsers:
                    linked = GuideParser()
                    linked.feed(target.read_text())
                    page_parsers[target] = linked
                assert fragment in page_parsers[target].ids, f"Missing HTML anchor: {page.name}: {href}"
            elif re.fullmatch(r"L\d+", fragment):
                assert 1 <= int(fragment[1:]) <= len(target.read_text().splitlines()), href
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
        call_stack_sections=len(stack_parser.trees),
        call_stack_source_targets=len(stack_manifest["references"]),
        memory_planning_source_targets=len(memory_manifest["references"]),
        memory_planning_complete_code_cells=len(memory_manifest["complete_code_cells"]),
        memory_planning_displayed_source_lines=memory_line_count,
        memory_planning_qna_cells=memory_parser.qna_cells,
        memory_planning_max_disclosure_depth=memory_parser.max_disclosure_depth,
        local_links_checked=checked_links, external_runtime_dependencies=0,
        evidence="Document/source validation only; no MAX model execution.",
    )
    (TASK / "validation").mkdir(exist_ok=True)
    (TASK / "validation" / "source_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
