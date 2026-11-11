"""Small offline renderer shared by new source-ordered example call trees."""
from __future__ import annotations

import html
import re

import markdown

TOKEN = re.compile(r"\[LINK:([\w-]+)(?::(\d+))?(?:\|([^\]]+))?\]|\[CALLER:([\w-]+)\|(\d+)\]|\[CARD:([^|]+)\|([^\]]+)\]|\[DETAIL:([^|]+)\|([^\]]+)\]")
BRANCH = re.compile(r"^(?P<prefix>[ │]*)(?:├─|└─)")


class CallTreeReader:
    def __init__(self, refs):
        self.refs = refs
        self.used_refs = set()
        self.caller_counts = {}
        self.source_counts = {}

    def source(self, key, line=None, label="↗", *, caller=False, tree_id=""):
        r = self.refs[key]
        line = r["line"] if line is None else int(line)
        if not r["start"] <= line <= r["end"]:
            raise ValueError(f"Call-tree source outside pinned span: {key}:{line}")
        self.used_refs.add(key)
        self.source_counts[tree_id] = self.source_counts.get(tree_id, 0) + 1
        classes = "call-source qna-source-path call-stack-source-link"
        extra = ""
        if caller:
            classes += " calltree-caller-link"
            extra = f' data-caller-source="true" data-caller-tree="{html.escape(tree_id, quote=True)}"'
            label = f"↗ caller [{r['alias']}:L{line}]"
        title = f"{'Caller · ' if caller else ''}{r['file']}:L{line}"
        return (f'<a class="{classes}" data-ref="{key}" data-line="{line}"{extra} '
                f'href="../sources/{html.escape(r["file"])}#L{line}" '
                f'title="{html.escape(title, quote=True)}" '
                f'aria-label="Open {html.escape(title, quote=True)}">{html.escape(label)}</a>')

    def decorate(self, text, tree_id=""):
        out, previous = [], 0
        for m in TOKEN.finditer(text):
            out.append(html.escape(text[previous:m.start()]))
            if m[1]:
                out.append(self.source(m[1], m[2], m[3] or "↗", tree_id=tree_id))
            elif m[4]:
                out.append(self.source(m[4], m[5], caller=True, tree_id=tree_id))
            elif m[6]:
                out.append(f'<a class="call-note-link" href="#{html.escape(m[6])}" '
                           f'data-call-explanation="{html.escape(m[6], quote=True)}">{html.escape(m[7])}</a>')
            else:
                out.append(f'<a class="call-note-link" href="{html.escape(m[8], quote=True)}">{html.escape(m[9])}</a>')
            previous = m.end()
        out.append(html.escape(text[previous:]))
        result = "".join(out)
        result = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", result)
        return re.sub(r"`([^`]+)`", r'<span class="call-stack-inline-code">\1</span>', result)

    def annotate(self, text, fallback_key, fallback_line):
        """All action/detail rows link to source; leaves also identify callers."""
        rows = text.strip().splitlines()
        stack = []
        for i, raw in enumerate(rows):
            match = BRANCH.match(raw)
            own = next((m for m in TOKEN.finditer(raw) if m[1]), None)
            if match:
                depth = len(match["prefix"]) // 3
                while stack and stack[-1][0] >= depth:
                    stack.pop()
                parent = stack[-1][1:] if stack else (fallback_key, fallback_line)
                following = next((BRANCH.match(row) for row in rows[i+1:] if BRANCH.match(row)), None)
                leaf = following is None or len(following["prefix"]) // 3 <= depth
                if own is None:
                    rows[i] += f" [LINK:{parent[0]}:{parent[1]}]"
                if leaf and "[CALLER:" not in raw:
                    rows[i] += f" [CALLER:{parent[0]}|{parent[1]}]"
                key = own[1] if own else parent[0]
                line = int(own[2]) if own and own[2] else self.refs[key]["line"] if own else parent[1]
                stack.append((depth, key, line))
            elif raw.strip(" │") and own is None:
                key, line = stack[-1][1:] if stack else (fallback_key, fallback_line)
                rows[i] += f" [LINK:{key}:{line}]"
        return "\n".join(rows)

    def tree(self, text, tree_id, title, fallback_key, fallback_line):
        marked = self.annotate(text, fallback_key, fallback_line)
        self.caller_counts[tree_id] = len(re.findall(r"\[CALLER:", marked))
        rows = []
        for row in marked.splitlines():
            copy = TOKEN.sub(lambda m: (m[3] or "↗") if m[1] else
                             f"↗ caller [L{m[5]}]" if m[4] else m[7] if m[6] else m[9], row)
            rows.append('<span class="call-stack-copy-line" '
                        f'data-copy-line="{html.escape(copy, quote=True)}">'
                        f'<span class="call-stack-line-text">{self.decorate(row, tree_id)}</span></span>')
        return (f'<div class="call-tree-scroll call-stack-copy-tree" id="{tree_id}" '
                f'data-call-tree-id="{tree_id}" tabindex="0" role="region" '
                f'aria-label="{html.escape(title, quote=True)}; scroll horizontally for long branches">'
                '<pre class="call-tree"><code>' + "\n".join(rows) + '</code></pre></div>')

    def explanation(self, text):
        anchors = {}

        def source_anchor(m):
            file, line = m[2], int(m[3])
            choices = [k for k, r in self.refs.items()
                       if r["file"] == file and r["start"] <= line <= r["end"]]
            if not choices:
                raise ValueError(f"Explanation link has no pinned span: {file}:{line}")
            key = min(choices, key=lambda k: self.refs[k]["end"] - self.refs[k]["start"])
            token = f"SOURCEANCHOR{len(anchors)}TOKEN"
            anchors[token] = self.source(key, line, m[1])
            return token

        text = re.sub(r"\[([^\]]+)\]\(sources/([^\s)]+)#L(\d+)\)", source_anchor, text)
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
        for token, anchor in anchors.items():
            rendered = rendered.replace(token, anchor)
        rendered = re.sub(r"(<table>.*?</table>)", r'<div class="qna-table-scroll" tabindex="0">\1</div>', rendered, flags=re.S)
        return '<div class="call-explanation">' + rendered + '</div>'


EXPLANATION_WINDOW = '''<aside class="call-explanation-card" id="call-explanation-card" hidden role="dialog" aria-modal="false" aria-labelledby="call-explanation-title"><div class="call-explanation-card-header"><h2 id="call-explanation-title">Explanation</h2><button type="button" data-close-call-explanation aria-label="Close explanation">Close ×</button></div><div class="call-explanation-card-body"></div></aside>'''
