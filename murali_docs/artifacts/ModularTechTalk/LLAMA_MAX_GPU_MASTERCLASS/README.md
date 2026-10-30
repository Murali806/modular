# Llama / MAX / GPU Masterclass

A source-driven course covering MAX graph construction, graph compiler transformations, Mojo GPU kernels, driver offload, target backends, and MAX Serve scheduling.

- PowerPoint: `decks/` (546 slides across six files)
- Browser course: `html/index.html`
- Searchable source index: `html/source-index.html`
- Source/function map: `COURSE_MAP.md`
- Displayed source line explanations: `LINE_BY_LINE_SOURCE_ATLAS.md`
- Reproducible generator: `src/`

## Build

```bash
cd murali_docs/artifacts/ModularTechTalk/LLAMA_MAX_GPU_MASTERCLASS/src
npm install
npm run build
npm run check
```

Revision used for source excerpts: `eab325f81bdbcf0195c511a56510f65c37540798`. GPU timing and achieved overlap remain `GPU-LAB` items; source inspection is not presented as measured GPU performance.
