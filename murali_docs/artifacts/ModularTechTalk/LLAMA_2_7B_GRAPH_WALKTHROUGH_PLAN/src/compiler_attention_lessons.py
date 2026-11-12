"""Source-ordered, card-specific lessons; no compiler or device execution."""
from __future__ import annotations

from copy import deepcopy

T_GPU = "float32[32,32] @ gpu:0"
T_CPU = "float32[8,8] @ cpu:0"
NAME = "modular_ops::fused_attention_custom"


def objects(*items):
    """Named objects, kinds and owners are stable across a lesson's frames."""
    return [dict(id=item[0], label=item[1], kind=item[2], owner=item[3])
            for item in items]


class Story:
    def __init__(self, nodes, *, label="Actual GPU caller", note="", view="objects"):
        self.nodes = nodes
        self.label = label
        self.note = note
        self.view = view
        self.values = {n["id"]: None for n in nodes}
        self.scene = {}
        self.frames = []

    def step(self, title, source, line, input_, action, output, why, *,
             patch=None, focus=(), phase="", **scene):
        previous = deepcopy(self.values)
        self.values.update(patch or {})
        self.scene.update(scene)
        frame = dict(title=title, refs=[[source, line]], input=input_, action=action,
                     output=output, why=why, detail=why, phase=phase,
                     fields={k: v for k, v in self.values.items() if v is not None},
                     values=deepcopy(self.values), before_values=previous,
                     focus=list(focus or (patch or {}).keys()),
                     before=input_, after=output, scene=deepcopy(self.scene))
        self.frames.append(frame)
        return self

    def finish(self):
        return dict(label=self.label, note=self.note, view=self.view,
                    nodes=self.nodes, frames=self.frames)


def lesson(key, title, view, question, inputs, outputs, explanation, tree,
           story, *, alternatives=None, source="attention_callsite", line=41):
    scenarios = {"actual": story.finish()}
    scenarios.update({k: v.finish() for k, v in (alternatives or {}).items()})
    return dict(id="toolchain-call-" + key, key=key, title=title, mechanism=key,
                view=view, question=question, inputs=inputs, outputs=outputs,
                explanation=explanation, tree=tree, scenarios=scenarios,
                source=source, line=line)


def all_lessons():
    from compiler_attention_graph_lessons import graph_lessons
    from compiler_attention_runtime_lessons import runtime_lessons
    from compiler_attention_kernel_lessons import kernel_lessons
    lessons = graph_lessons() + runtime_lessons() + kernel_lessons()
    ids = [item["id"] for item in lessons]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate attention card identity")
    for item in lessons:
        if not item["tree"] or not item["explanation"]:
            raise ValueError("Attention lesson needs a source tree and explanation")
        for scenario in item["scenarios"].values():
            if len(scenario["frames"]) < 2:
                raise ValueError("Attention lesson needs observable transitions")
    return lessons
