"""Small authoring vocabulary: explicit objects, source anchors, state changes."""
CHAPTERS = []


def ref(file, needle, label=None, occurrence=1):
    return dict(file=file, needle=needle, label=label, occurrence=occurrence)


def actor(id, title, kind, owner, purpose):
    return dict(id=id, title=title, kind=kind, owner=owner, purpose=purpose)


def item(id, label):
    return dict(id=id, label=label)


def box(state, *slots, facts=()):
    return dict(state=state, slots=[item(*s) if isinstance(s, tuple) else item(s, s) for s in slots], facts=list(facts))


def step(title, detail, changes, refs, *, transfer=None, edges=()):
    return dict(title=title, detail=detail, changes=changes, refs=refs,
                transfer=transfer, edges=list(edges))


def section(title, text, refs=(), *, open=False):
    return dict(title=title, text=text.strip(), refs=list(refs), open=open)


def chapter(slug, title, subtitle, group, actors, steps, sections, tree, refs, *, lab=None, scenarios=None):
    obj = dict(slug=slug, title=title, subtitle=subtitle, group=group, actors=actors,
               steps=steps, sections=sections, tree=tree.strip(), refs=refs,
               lab=lab, scenarios=scenarios or [])
    CHAPTERS.append(obj)
    return obj
