# Llama / MAX / GPU Masterclass

A source-driven course covering MAX graph construction, graph compiler transformations, Mojo GPU kernels, driver offload, target backends, and MAX Serve scheduling. The final interface follows the approved V4 synchronized-laboratory model.

- PowerPoint: `decks/` (586 slides across six files)
- Browser course: `html/index.html`
- Searchable source index: `html/source-index.html`
- Source/function map: `COURSE_MAP.md`
- Displayed source line explanations: `LINE_BY_LINE_SOURCE_ATLAS.md`
- Animation inventory: `animation_manifest.json`
- Reproducible generator: `src/`

## Build

```bash
cd murali_docs/artifacts/ModularTechTalk/LLAMA_MAX_GPU_MASTERCLASS/src
npm install
npm run build
npm run check
```

Revision used for source excerpts: `e9818b79ea73b498ec4b438a24ecd8d853307d0d`. The decks contain 568 animated content slides and 1704 On Click groups. GPU timing, occupancy, achieved overlap, and bandwidth remain `GPU-LAB` items; source inspection is not presented as measured device performance.
