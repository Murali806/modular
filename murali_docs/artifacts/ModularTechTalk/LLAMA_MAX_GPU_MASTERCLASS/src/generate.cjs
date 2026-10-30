#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const PptxGenJS = require("pptxgenjs");
const { parts } = require("./curriculum.cjs");
const { injectAnimations } = require("./animate.cjs");

const ROOT = path.resolve(__dirname, "../../../../..");
const OUT = path.resolve(__dirname, "..");
const DECKS = path.join(OUT, "decks");
const HTML = path.join(OUT, "html");
const ASSETS = path.join(HTML, "assets");
const REV = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
const SHAPE = new PptxGenJS().ShapeType;

const C = {
  paper: "F2F0EA", panel: "FBFAF7", white: "FFFFFF", ink: "171C1E",
  dark: "172023", dark2: "222D31", muted: "687277", line: "CBC7BD",
  lineDark: "99A1A3", source: "087D70", boundary: "B64A35",
  compile: "7053A3", gpu: "246FA2", orange: "C25E25", yellow: "D39C20",
};

const ensure = (dir) => fs.mkdirSync(dir, { recursive: true });
const esc = (value) => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const clip = (value, length = 95) => String(value).length > length
  ? `${String(value).slice(0, length - 3)}...` : String(value);
const slugify = (value) => String(value).toLowerCase()
  .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const animName = (slide, group, suffix) => `anim-s${String(slide).padStart(3, "0")}-g${String(group).padStart(2, "0")}-${slugify(suffix)}`;

function readSource(file) {
  const full = path.join(ROOT, file);
  if (!fs.existsSync(full)) throw new Error(`Missing source file: ${file}`);
  return fs.readFileSync(full, "utf8").replaceAll("\t", "    ").split(/\r?\n/);
}

function sourceWindow(unit, half) {
  const lines = readSource(unit.file);
  const joined = lines.join("\n");
  const offset = joined.indexOf(unit.anchor);
  if (offset < 0) throw new Error(`Anchor not found: ${unit.file} :: ${unit.anchor}`);
  const anchorLine = joined.slice(0, offset).split("\n").length - 1;
  const base = Math.max(0, Math.min(lines.length - 16, anchorLine - 1));
  let start = base;
  if (half === 1) {
    const terms = [
      "addPass", "buildStage", "return", "enqueue_", "compile", "verify",
      "claim", "release", "output", "create", "copy", "fuse", "budget",
      "cache", "device", "memory", "load", "execute", "append", "rewrite",
      "convert", "alloc", "stream", "kernel", "model", "graph", "buffer",
    ];
    const searchStart = Math.min(lines.length - 8, anchorLine + 8);
    const searchEnd = Math.min(lines.length - 8, anchorLine + 180);
    let bestScore = -1;
    let bestStart = searchStart;
    for (let candidate = searchStart; candidate <= searchEnd; candidate += 1) {
      const window = lines.slice(candidate, candidate + 8);
      let score = window.reduce((sum, text) => {
        const trimmed = text.trim();
        let lineScore = trimmed ? 1 : -0.5;
        for (const term of terms) if (text.includes(term)) lineScore += 3;
        if (/^(if|for|while|return|comptime if)\b/.test(trimmed)) lineScore += 2;
        if (trimmed.startsWith("#") || trimmed.startsWith("//")) lineScore += 0.5;
        return sum + lineScore;
      }, 0);
      if (candidate < anchorLine + 16) score -= 2;
      if (score > bestScore) {
        bestScore = score;
        bestStart = candidate;
      }
    }
    start = bestStart;
  }
  return Array.from({ length: 8 }, (_, index) => ({
    number: start + index + 1,
    text: lines[start + index] === undefined ? "" : lines[start + index],
  }));
}

function explainLine(text) {
  const value = text.trim();
  if (!value) return "This blank line separates logical source blocks; it performs no runtime work.";
  if (value.startsWith("#") || value.startsWith("//")) return "The source comment records an invariant, intent, target distinction, or implementation boundary.";
  if (value.startsWith("@")) return "The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.";
  if (/^(async\s+)?def\s/.test(value) || /^[\w:<>,*&\s]+\([^;]*\)\s*\{?$/.test(value)) return "This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.";
  if (/^(class|struct)\s/.test(value)) return "This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.";
  if (/^(if|elif|else|switch|case|comptime if)\b/.test(value)) return "This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.";
  if (/^(for|while|comptime for)\b/.test(value)) return "This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.";
  if (value.includes("addPass") || value.includes("buildStage")) return "This fixes transformation order. The next pass consumes invariants produced by the pass registered here.";
  if (value.includes("enqueue_")) return "This enqueues asynchronous work. Host completion is not implied until a dependency or explicit synchronization requires it.";
  if (value.includes("verify")) return "This validates the current object or IR contract and surfaces diagnostics close to the source that created it.";
  if (value.startsWith("return")) return "The returned object becomes state consumed by the caller or the next compiler/runtime stage.";
  if (value.includes("=")) return "This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.";
  return "This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.";
}

function buildScenes(part) {
  const scenes = [];
  part.units.forEach((unit, unitIndex) => {
    for (let half = 0; half < 2; half += 1) {
      const lines = sourceWindow(unit, half);
      scenes.push({
        id: `p${part.id}-${String(unitIndex + 1).padStart(2, "0")}-${half + 1}`,
        number: scenes.length + 1,
        part: part.id,
        title: `${unit.title} ${half === 0 ? "I - contract" : "II - implementation"}`,
        unitTitle: unit.title,
        file: unit.file,
        anchor: unit.anchor,
        role: unit.role,
        concepts: unit.concepts,
        diagram: unit.diagram,
        evidence: unit.evidence,
        formula: unit.formula,
        lines,
        explanations: lines.map((line) => explainLine(line.text)),
      });
    }
  });
  return scenes;
}

const course = parts.map((part) => ({ ...part, scenes: buildScenes(part) }));

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: options.mono ? "Aptos Mono" : "Aptos",
    fontSize: options.size || 10,
    color: options.color || C.ink,
    bold: !!options.bold,
    margin: options.margin === undefined ? 0 : options.margin,
    valign: options.valign || "mid",
    align: options.align || "left",
    fit: "shrink",
    breakLine: false,
    paraSpaceAfterPt: 0,
    objectName: options.name,
    ...options.extra,
  });
}

function shape(slide, type, x, y, w, h, options = {}) {
  const transparent = { color: C.white, transparency: 100 };
  slide.addShape(type, {
    x, y, w, h,
    fill: options.fill === null ? transparent : { color: options.fill || C.white, transparency: options.transparency || 0 },
    line: options.line === null ? transparent : {
      color: options.line || options.fill || C.line,
      width: options.width || 1,
      dash: options.dash || "solid",
      beginArrowType: options.beginArrowType || "none",
      endArrowType: options.endArrowType || "none",
    },
    objectName: options.name,
    radius: options.radius,
    ...options.extra,
  });
}

function notes(slide, value) {
  if (typeof slide.addNotes === "function") slide.addNotes(value);
}

function evidenceColor(scene) {
  if (scene.evidence.includes("BOUNDARY")) return C.boundary;
  if (scene.evidence.includes("GPU-LAB")) return C.gpu;
  if (scene.evidence.includes("COMPILE-ONLY")) return C.compile;
  return C.source;
}

function slideHeader(slide, part, scene, build, slideNo, total) {
  shape(slide, SHAPE.rect, 0, 0, 13.333, 0.76, { fill: C.panel, line: C.line });
  shape(slide, SHAPE.rect, 0, 0, 0.12, 0.76, { fill: part.color, line: part.color });
  addText(slide, `PART ${String(part.id).padStart(2, "0")} / ${scene.diagram.toUpperCase()}`, 0.28, 0.1, 2.4, 0.18, { size: 7.5, color: part.color, bold: true });
  addText(slide, scene.title, 0.28, 0.3, 7.85, 0.31, { size: 17, bold: true });
  addText(slide, clip(scene.role, 115), 8.25, 0.17, 3.45, 0.38, { size: 7.3, color: C.muted, valign: "top" });
  shape(slide, SHAPE.roundRect, 11.85, 0.13, 0.72, 0.28, { fill: C.white, line: evidenceColor(scene), radius: 0.03 });
  addText(slide, clip(scene.evidence, 14), 11.88, 0.15, 0.66, 0.23, { size: 5.7, color: evidenceColor(scene), bold: true, align: "center" });
  addText(slide, `${build}/4`, 12.68, 0.15, 0.42, 0.23, { size: 8.5, color: part.color, bold: true, mono: true, align: "right" });
  addText(slide, `${slideNo}/${total}`, 12.62, 0.47, 0.48, 0.14, { size: 6.2, color: C.muted, mono: true, align: "right" });
}

function sourcePanel(slide, scene, build, part, slideNo) {
  const x = 0.16, y = 0.86, w = 4.37, h = 6.08;
  shape(slide, SHAPE.rect, x, y, w, h, { fill: C.dark, line: C.lineDark });
  shape(slide, SHAPE.rect, x, y, w, 0.37, { fill: C.dark2, line: C.dark2 });
  addText(slide, clip(scene.file, 73), x + 0.13, y + 0.08, 3.48, 0.18, { size: 6.5, color: "AAB5B8", mono: true });
  addText(slide, `L${scene.lines[0].number}-L${scene.lines.at(-1).number}`, x + 3.62, y + 0.08, 0.58, 0.18, { size: 6.1, color: "AAB5B8", mono: true, align: "right" });
  const visible = build * 2;
  scene.lines.forEach((entry, index) => {
    const lineY = y + 0.47 + index * 0.53;
    const shown = index < visible;
    const current = index >= visible - 2 && index < visible;
    shape(slide, SHAPE.rect, x, lineY, w, 0.5, {
      fill: current ? "2B3A3E" : shown ? (index % 2 ? "192225" : C.dark) : "13191B",
      line: null,
      transparency: shown ? 0 : 12,
      name: current ? animName(slideNo, 1, `source-bg-${index}`) : undefined,
    });
    if (current) shape(slide, SHAPE.rect, x, lineY, 0.04, 0.5, { fill: part.color, line: part.color, name: animName(slideNo, 1, `source-bar-${index}`) });
    addText(slide, String(entry.number), x + 0.08, lineY + 0.11, 0.36, 0.19, {
      size: 6.5, color: shown ? "768489" : "3B474B", mono: true, align: "right",
      name: current ? animName(slideNo, 1, `source-number-${index}`) : undefined,
    });
    addText(slide, shown ? clip(entry.text, 70) : "", x + 0.56, lineY + 0.08, 3.65, 0.28, {
      size: 7.15, color: current ? C.white : shown ? "D8DEDF" : "4C585C", mono: true,
      name: current ? animName(slideNo, 1, `source-text-${index}`) : undefined,
    });
  });
  const focusIndex = visible - 1;
  const focus = scene.lines[focusIndex];
  shape(slide, SHAPE.rect, x, y + 4.83, w, 1.25, { fill: "F1EFE9", line: C.line });
  addText(slide, `L${focus.number}`, x + 0.16, y + 4.98, 0.43, 0.18, { size: 6.8, color: part.color, bold: true, mono: true });
  addText(slide, scene.explanations[focusIndex], x + 0.68, y + 4.91, 3.48, 0.74, { size: 7.1, color: C.ink, valign: "top" });
  addText(slide, `ANCHOR  ${clip(scene.anchor.replaceAll("\n", " "), 58)}`, x + 0.16, y + 5.78, 4.03, 0.16, { size: 5.8, color: C.muted, mono: true });
}

function stepChips(slide, scene, build, part) {
  scene.concepts.forEach((concept, index) => {
    const chipX = 4.78 + index * 1.28;
    const active = index === build - 1;
    const done = index < build;
    shape(slide, SHAPE.roundRect, chipX, 0.98, 1.2, 0.32, {
      fill: active ? part.color : done ? "EDF6F3" : C.white,
      line: done ? part.color : C.line,
      radius: 0.03,
    });
    addText(slide, clip(concept, 18), chipX + 0.05, 1.02, 1.1, 0.21, { size: 5.8, color: active ? C.white : done ? part.color : C.muted, bold: done, align: "center" });
  });
}

function conceptBox(slide, text, x, y, w, h, part, status, slideNo, suffix) {
  const current = status === "current";
  const revealed = current || status === "done";
  shape(slide, SHAPE.roundRect, x, y, w, h, {
    fill: current ? "EAF6F2" : revealed ? C.white : "F3F1EC",
    line: revealed ? part.color : C.line,
    transparency: revealed ? 0 : 18,
    radius: 0.04,
    name: current ? animName(slideNo, 2, `${suffix}-box`) : undefined,
  });
  addText(slide, text, x + 0.09, y + 0.08, w - 0.18, h - 0.16, {
    size: 8, color: revealed ? C.ink : C.muted, bold: revealed, align: "center",
    name: current ? animName(slideNo, 2, `${suffix}-text`) : undefined,
  });
}

function flowDiagram(slide, scene, build, part, slideNo, vertical = false, boundary = false) {
  scene.concepts.forEach((concept, index) => {
    const x = vertical ? 6.15 : 4.93 + (index % 2) * 2.48;
    const y = vertical ? 1.65 + index * 1.15 : 1.75 + Math.floor(index / 2) * 2.0;
    const status = index === build - 1 ? "current" : index < build ? "done" : "future";
    conceptBox(slide, concept, x, y, vertical ? 2.65 : 2.2, vertical ? 0.74 : 1.18, part, status, slideNo, `flow-${index}`);
    if (boundary) shape(slide, SHAPE.line, x + 0.05, y + (vertical ? 0.63 : 1.07), vertical ? 2.55 : 2.1, 0, { line: C.boundary, dash: "dash" });
    if (index > 0 && index < build) {
      const current = index === build - 1;
      if (vertical) shape(slide, SHAPE.line, 7.47, y - 0.35, 0, 0.32, { line: part.color, endArrowType: "triangle", width: 1.6, name: current ? animName(slideNo, 2, `flow-arrow-${index}`) : undefined });
      else if (index % 2 === 1) shape(slide, SHAPE.line, x - 0.24, y + 0.58, 0.2, 0, { line: part.color, endArrowType: "triangle", width: 1.6, name: current ? animName(slideNo, 2, `flow-arrow-${index}`) : undefined });
      else shape(slide, SHAPE.line, x + 1.1, y - 0.72, 0, 0.68, { line: part.color, endArrowType: "triangle", width: 1.6, name: current ? animName(slideNo, 2, `flow-arrow-${index}`) : undefined });
    }
  });
}

function irDiagram(slide, scene, build, part, slideNo) {
  addText(slide, "BEFORE", 4.92, 1.52, 1.2, 0.18, { size: 6.4, color: C.muted, bold: true });
  addText(slide, "AFTER", 8.45, 1.52, 1.2, 0.18, { size: 6.4, color: part.color, bold: true });
  scene.concepts.forEach((concept, index) => {
    const x = index < 2 ? 4.93 : 8.2;
    const y = 1.88 + (index % 2) * 1.35;
    const status = index === build - 1 ? "current" : index < build ? "done" : "future";
    conceptBox(slide, `%${index}  ${concept}`, x, y, 1.78, 0.86, part, status, slideNo, `ir-${index}`);
  });
  shape(slide, SHAPE.chevron, 6.96, 2.32, 0.92, 1.12, { fill: build >= 3 ? part.color : "DEDAD1", line: build >= 3 ? part.color : C.line, name: build === 3 ? animName(slideNo, 2, "ir-pass") : undefined });
  addText(slide, "PASS", 7.05, 2.7, 0.68, 0.18, { size: 7, color: build >= 3 ? C.white : C.muted, bold: true, align: "center", name: build === 3 ? animName(slideNo, 2, "ir-pass-text") : undefined });
}

function memoryDiagram(slide, scene, build, part, slideNo) {
  addText(slide, "OWNER / SOURCE", 4.95, 1.53, 1.5, 0.18, { size: 6.2, color: C.muted, bold: true });
  addText(slide, "NEXT RESIDENCE", 8.5, 1.53, 1.4, 0.18, { size: 6.2, color: part.color, bold: true });
  scene.concepts.forEach((concept, index) => {
    const y = 1.84 + index * 0.94;
    const current = index === build - 1;
    const revealed = index < build;
    conceptBox(slide, concept, 4.94, y, 1.78, 0.57, part, current ? "current" : revealed ? "done" : "future", slideNo, `mem-left-${index}`);
    conceptBox(slide, index === 3 ? "consumer result" : scene.concepts[Math.min(3, index + 1)], 8.28, y, 1.78, 0.57, part, current ? "current" : revealed ? "done" : "future", slideNo, `mem-right-${index}`);
    if (revealed) {
      shape(slide, SHAPE.line, 6.9, y + 0.29, 1.18, 0, { line: part.color, endArrowType: "triangle", width: 1.8, name: current ? animName(slideNo, 2, `mem-arrow-${index}`) : undefined });
      if (current) shape(slide, SHAPE.ellipse, 7.1, y + 0.18, 0.2, 0.2, { fill: part.color, line: part.color, name: animName(slideNo, 2, `mem-packet-${index}`) });
    }
  });
}

function tileDiagram(slide, scene, build, part, slideNo) {
  const matrices = [{ x: 4.84, label: "A / INPUT" }, { x: 8.25, label: "B / OUTPUT" }];
  matrices.forEach((matrix, matrixIndex) => {
    shape(slide, SHAPE.rect, matrix.x, 1.62, 1.92, 3.25, { fill: C.white, line: C.lineDark });
    addText(slide, matrix.label, matrix.x + 0.12, 1.74, 1.6, 0.18, { size: 6.5, color: C.ink, bold: true, mono: true });
    for (let row = 0; row < 8; row += 1) {
      for (let col = 0; col < 8; col += 1) {
        const active = build > 0 && ((row + col + matrixIndex) % Math.max(2, 6 - build) === 0 || (row < build + 1 && col < build + 1));
        shape(slide, SHAPE.rect, matrix.x + 0.12 + col * 0.205, 2.08 + row * 0.31, 0.17, 0.26, { fill: active ? (matrixIndex ? "D9CFF3" : "7BC4AE") : "F0EEE8", line: active ? part.color : "D8D5CE", width: 0.5 });
      }
    }
  });
  scene.concepts.forEach((concept, index) => conceptBox(slide, concept, 6.99, 1.72 + index * 0.82, 1.04, 0.55, part, index === build - 1 ? "current" : index < build ? "done" : "future", slideNo, `tile-${index}`));
}

function timelineDiagram(slide, scene, build, part, slideNo) {
  const lanes = ["HOST", "COPY", "QUEUE", "GPU"];
  lanes.forEach((lane, index) => {
    const y = 1.72 + index * 0.91;
    addText(slide, lane, 4.88, y + 0.12, 0.62, 0.18, { size: 6.3, color: C.muted, bold: true, mono: true });
    shape(slide, SHAPE.line, 5.56, y + 0.24, 4.42, 0, { line: C.line, width: 0.8 });
    const current = index === build - 1;
    if (index < build) {
      shape(slide, SHAPE.roundRect, 5.72 + index * 0.55, y, 2.0 + (3 - index) * 0.28, 0.48, { fill: current ? "EAF6F2" : C.white, line: part.color, radius: 0.03, name: current ? animName(slideNo, 2, `timeline-${index}`) : undefined });
      addText(slide, scene.concepts[index], 5.82 + index * 0.55, y + 0.08, 1.8 + (3 - index) * 0.28, 0.26, { size: 6.5, bold: true, align: "center", name: current ? animName(slideNo, 2, `timeline-text-${index}`) : undefined });
    }
  });
  shape(slide, SHAPE.line, 5.68 + build * 0.85, 1.55, 0, 4.0, { line: part.color, width: 1.6, name: animName(slideNo, 2, "timeline-playhead") });
}

function rooflineDiagram(slide, scene, build, part, slideNo) {
  const x = 5.08, y = 5.37, w = 3.72, h = 3.35;
  shape(slide, SHAPE.line, x, y, w, 0, { line: C.ink, width: 1.2 });
  shape(slide, SHAPE.line, x, y, 0, -h, { line: C.ink, width: 1.2 });
  [0.8, 1.6, 2.4].forEach((offset) => shape(slide, SHAPE.line, x, y - offset, w, 0, { line: "DDD8CF", dash: "dash", width: 0.6 }));
  shape(slide, SHAPE.line, x + 0.08, y - 0.08, 2.35, -2.45, { line: part.color, width: 2.8 });
  shape(slide, SHAPE.line, x + 2.43, y - 2.53, 1.18, 0, { line: C.yellow, width: 2.8 });
  shape(slide, SHAPE.ellipse, x + 0.45 + build * 0.62, y - 0.38 - build * 0.56, 0.22, 0.22, { fill: C.orange, line: C.white, width: 1.2, name: animName(slideNo, 2, "roof-point") });
  addText(slide, "Arithmetic intensity ->", 5.75, 5.56, 2.7, 0.18, { size: 6.2, color: C.muted, mono: true, align: "center" });
  scene.concepts.forEach((concept, index) => conceptBox(slide, concept, 8.98, 1.65 + index * 0.88, 1.0, 0.58, part, index === build - 1 ? "current" : index < build ? "done" : "future", slideNo, `roof-${index}`));
}

function backendDiagram(slide, scene, build, part, slideNo) {
  conceptBox(slide, scene.concepts[0], 6.23, 1.65, 2.25, 0.78, part, build >= 1 ? (build === 1 ? "current" : "done") : "future", slideNo, "backend-core");
  scene.concepts.slice(1).forEach((concept, local) => {
    const index = local + 1;
    const x = local % 2 ? 7.55 : 4.96;
    const y = 3.08 + Math.floor(local / 2) * 1.2;
    conceptBox(slide, concept, x, y, 2.25, 0.76, part, index === build - 1 ? "current" : index < build ? "done" : "future", slideNo, `backend-${index}`);
    if (index < build) shape(slide, SHAPE.line, 7.34, 2.42, x + 1.12 - 7.34, y - 2.42, { line: part.color, endArrowType: "triangle", width: 1.4, dash: "dash", name: index === build - 1 ? animName(slideNo, 2, `backend-arrow-${index}`) : undefined });
  });
  shape(slide, SHAPE.rect, 5.15, 5.37, 4.62, 0.58, { fill: "FFF0EB", line: C.boundary, dash: "dash" });
  addText(slide, "NATIVE / VENDOR IMPLEMENTATION BOUNDARY", 5.28, 5.53, 4.35, 0.18, { size: 6.4, color: C.boundary, bold: true, align: "center" });
}

function stateDiagram(slide, scene, build, part, slideNo) {
  scene.concepts.forEach((concept, index) => {
    const x = 5.0 + (index % 2) * 2.52;
    const y = 1.72 + Math.floor(index / 2) * 2.0;
    const status = index === build - 1 ? "current" : index < build ? "done" : "future";
    conceptBox(slide, concept, x, y, 2.17, 1.22, part, status, slideNo, `state-${index}`);
    shape(slide, SHAPE.ellipse, x + 0.12, y - 0.14, 0.38, 0.38, { fill: status === "current" ? part.color : C.white, line: part.color, name: status === "current" ? animName(slideNo, 2, `state-index-${index}`) : undefined });
    addText(slide, String(index + 1), x + 0.12, y - 0.12, 0.38, 0.3, { size: 6.6, color: status === "current" ? C.white : part.color, bold: true, align: "center", name: status === "current" ? animName(slideNo, 2, `state-index-text-${index}`) : undefined });
  });
}

function diagramPanel(slide, scene, build, part, slideNo) {
  const x = 4.64, y = 0.86, w = 5.62, h = 6.08;
  shape(slide, SHAPE.rect, x, y, w, h, { fill: "FAF9F6", line: C.line });
  shape(slide, SHAPE.rect, x, y, w, 0.48, { fill: "F6F4EF", line: C.line });
  addText(slide, scene.diagram.toUpperCase(), x + 0.14, y + 0.12, 0.82, 0.18, { size: 6.2, color: part.color, bold: true });
  addText(slide, scene.unitTitle, x + 1.02, y + 0.1, 2.78, 0.22, { size: 8.3, bold: true });
  stepChips(slide, scene, build, part);
  if (scene.diagram === "ir") irDiagram(slide, scene, build, part, slideNo);
  else if (["memory", "buffers", "cache", "ownership"].includes(scene.diagram)) memoryDiagram(slide, scene, build, part, slideNo);
  else if (scene.diagram === "tile") tileDiagram(slide, scene, build, part, slideNo);
  else if (["timeline", "schedule", "queue"].includes(scene.diagram)) timelineDiagram(slide, scene, build, part, slideNo);
  else if (scene.diagram === "roofline") rooflineDiagram(slide, scene, build, part, slideNo);
  else if (["backend", "dispatch"].includes(scene.diagram)) backendDiagram(slide, scene, build, part, slideNo);
  else if (["state", "tree"].includes(scene.diagram)) stateDiagram(slide, scene, build, part, slideNo);
  else flowDiagram(slide, scene, build, part, slideNo, scene.diagram === "sequence", scene.diagram === "boundary");
}

function inspectorPanel(slide, scene, build, part, slideNo) {
  const x = 10.37, y = 0.86, w = 2.8, h = 6.08;
  const focusIndex = build * 2 - 1;
  const focus = scene.lines[focusIndex];
  shape(slide, SHAPE.rect, x, y, w, h, { fill: C.panel, line: C.line });
  shape(slide, SHAPE.rect, x, y, w, 0.48, { fill: C.white, line: C.line });
  addText(slide, `STATE ${String(build).padStart(2, "0")}`, x + 0.14, y + 0.12, 0.82, 0.18, { size: 6.3, color: part.color, bold: true, mono: true });
  addText(slide, scene.concepts[build - 1], x + 0.92, y + 0.1, 1.7, 0.22, { size: 8, bold: true, align: "right" });
  shape(slide, SHAPE.rect, x + 0.12, y + 0.62, w - 0.24, 0.82, { fill: "F3F1EC", line: part.color, name: animName(slideNo, 3, "inspector-summary") });
  addText(slide, scene.explanations[focusIndex], x + 0.23, y + 0.72, w - 0.46, 0.58, { size: 6.8, valign: "top", name: animName(slideNo, 3, "inspector-summary-text") });
  addText(slide, "OBJECT STATE", x + 0.14, y + 1.62, 1.1, 0.16, { size: 6.1, color: C.muted, bold: true });
  scene.concepts.forEach((concept, index) => {
    const rowY = y + 1.86 + index * 0.44;
    shape(slide, SHAPE.line, x + 0.13, rowY + 0.35, w - 0.26, 0, { line: "DDD9D0", width: 0.5 });
    addText(slide, clip(concept, 25), x + 0.16, rowY, 1.62, 0.25, { size: 6.4, color: index === build - 1 ? part.color : C.ink, bold: index < build });
    addText(slide, index < build ? "materialized" : index === build ? "next" : "pending", x + 1.82, rowY, 0.78, 0.25, { size: 5.7, color: C.muted, align: "right" });
  });
  addText(slide, "MATH / INVARIANT", x + 0.14, y + 3.78, 1.2, 0.16, { size: 6.1, color: C.muted, bold: true });
  shape(slide, SHAPE.rect, x + 0.12, y + 4.02, w - 0.24, 0.68, { fill: C.dark, line: part.color, name: animName(slideNo, 3, "inspector-formula") });
  addText(slide, scene.formula, x + 0.22, y + 4.1, w - 0.44, 0.48, { size: 6.3, color: C.white, mono: true, align: "center", name: animName(slideNo, 3, "inspector-formula-text") });
  addText(slide, "EVIDENCE BOUNDARY", x + 0.14, y + 4.9, 1.25, 0.16, { size: 6.1, color: C.muted, bold: true });
  const boundaryText = scene.evidence.includes("GPU-LAB")
    ? "Structure is source-visible. Timing, traffic, occupancy, and achieved overlap require a target GPU measurement."
    : scene.evidence.includes("BOUNDARY")
      ? "The handoff is source-visible; native or vendor implementation details remain below the marked boundary."
      : "The current claim is proven by the displayed repository lines.";
  addText(slide, boundaryText, x + 0.14, y + 5.12, w - 0.28, 0.58, { size: 6.2, color: C.muted, valign: "top" });
  addText(slide, `FOCUS L${focus.number}`, x + 0.14, y + 5.78, 1.0, 0.15, { size: 5.8, color: part.color, bold: true, mono: true });
}

function contentSlide(pptx, part, scene, build, slideNo, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  slideHeader(slide, part, scene, build, slideNo, total);
  sourcePanel(slide, scene, build, part, slideNo);
  diagramPanel(slide, scene, build, part, slideNo);
  inspectorPanel(slide, scene, build, part, slideNo);
  addText(slide, `${REV.slice(0, 12)}  |  ${scene.file}:${scene.lines[0].number}`, 0.2, 7.14, 8.6, 0.14, { size: 5.8, color: C.muted, mono: true });
  addText(slide, "CLICK 1: SOURCE  ·  CLICK 2: SYSTEM  ·  CLICK 3: STATE", 8.85, 7.14, 4.28, 0.14, { size: 5.8, color: part.color, bold: true, align: "right" });
  const visible = scene.lines.slice(0, build * 2);
  notes(slide, [
    scene.title, "", scene.role, "", `Evidence: ${scene.evidence}`,
    `Source: ${scene.file}:${scene.lines[0].number}-${scene.lines.at(-1).number}`,
    `Function/anchor: ${scene.anchor.replaceAll("\n", " ")}`, "", "Click sequence:",
    "1. Reveal the two source lines introduced by this build.",
    `2. Reveal the system object or transformation: ${scene.concepts[build - 1]}.`,
    "3. Reveal the object-state explanation and mathematical invariant.", "",
    ...visible.map((entry, index) => `L${entry.number}: ${entry.text.trim()}\n  ${scene.explanations[index]}`),
    "", `Mathematical model: ${scene.formula}`, "",
    "GPU-LAB statements require measurement on the selected target. Source inspection is not presented as measured device performance.",
  ].join("\n"));
}

function titleSlide(pptx, part, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  shape(slide, SHAPE.rect, 0.62, 0.55, 0.14, 6.18, { fill: part.color, line: part.color });
  addText(slide, `LLAMA / MAX / GPU MASTERCLASS  ·  PART ${String(part.id).padStart(2, "0")}`, 1.02, 0.72, 7.4, 0.28, { size: 10, color: part.color, bold: true });
  addText(slide, part.title, 1.0, 1.25, 7.35, 1.15, { size: 34, color: C.white, bold: true, valign: "top" });
  addText(slide, part.subtitle, 1.03, 2.67, 6.8, 0.82, { size: 15, color: "AAB5B8", valign: "top" });
  addText(slide, `${part.units.length} source units  ·  ${part.scenes.length} source scenes  ·  ${total} slides`, 1.03, 5.62, 6.5, 0.28, { size: 11, color: C.yellow, bold: true });
  addText(slide, `SOURCE REV ${REV.slice(0, 12)}`, 1.03, 6.13, 4.3, 0.22, { size: 8, color: "839094", mono: true });
  part.units.slice(0, 5).forEach((unit, index) => conceptBox(slide, unit.title, 8.72, 0.88 + index * 1.1, 3.72, 0.7, part, "done", 0, `title-${index}`));
  notes(slide, `${part.title}\n\n${part.subtitle}\n\nThis deck uses the approved synchronized source/system/state teaching grammar. Content slides contain three On Click groups.`);
}

function routeSlide(pptx, part, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addText(slide, `PART ${part.id} / SOURCE ROUTE`, 0.55, 0.35, 3.7, 0.22, { size: 8, color: part.color, bold: true });
  addText(slide, "Objects and functions in execution order", 0.55, 0.67, 8.8, 0.48, { size: 25, bold: true });
  addText(slide, `${part.units.length} functions or source anchors · ${part.scenes.length} exact source windows · ${total - 3} animated build slides`, 0.57, 1.18, 9.8, 0.27, { size: 8.5, color: C.muted });
  const columns = part.units.length > 12 ? 3 : 2;
  const rows = Math.ceil(part.units.length / columns);
  part.units.forEach((unit, index) => {
    const col = Math.floor(index / rows);
    const row = index % rows;
    const x = 0.62 + col * (12.1 / columns);
    const y = 1.72 + row * (4.95 / rows);
    shape(slide, SHAPE.ellipse, x, y, 0.35, 0.35, { fill: part.color, line: part.color });
    addText(slide, String(index + 1), x, y + 0.02, 0.35, 0.26, { size: 6.5, color: C.white, bold: true, align: "center" });
    addText(slide, unit.title, x + 0.48, y - 0.02, 3.05, 0.19, { size: 7.3, bold: true });
    addText(slide, clip(unit.anchor.replaceAll("\n", " "), 46), x + 0.48, y + 0.18, 3.05, 0.15, { size: 5.7, color: C.muted, mono: true });
  });
  notes(slide, `Source route for ${part.title}. Each unit becomes a contract scene and implementation scene; each scene becomes four build slides.`);
}

function sourceIndexSlide(pptx, part) {
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  addText(slide, `PART ${part.id} / SOURCE INDEX`, 0.55, 0.35, 3.7, 0.22, { size: 8, color: part.color, bold: true });
  addText(slide, "Files, functions, evidence, and mathematical models", 0.55, 0.67, 10.3, 0.48, { size: 24, bold: true });
  const columns = part.units.length > 12 ? 3 : 2;
  const rows = Math.ceil(part.units.length / columns);
  part.units.forEach((unit, index) => {
    const col = Math.floor(index / rows);
    const row = index % rows;
    const cellW = 12.1 / columns;
    const x = 0.62 + col * cellW;
    const y = 1.48 + row * (5.55 / rows);
    addText(slide, `${String(index + 1).padStart(2, "0")}  ${unit.title}`, x, y, cellW - 0.28, 0.18, { size: 6.8, color: part.color, bold: true });
    addText(slide, clip(unit.file, columns === 3 ? 48 : 70), x, y + 0.19, cellW - 0.28, 0.14, { size: 5.3, color: C.muted, mono: true });
    addText(slide, `${clip(unit.evidence, 22)}  ·  ${clip(unit.formula, columns === 3 ? 34 : 52)}`, x, y + 0.34, cellW - 0.28, 0.15, { size: 5.2, color: C.ink, mono: true });
    shape(slide, SHAPE.line, x, y + 0.54, cellW - 0.28, 0, { line: C.line, width: 0.5 });
  });
  notes(slide, `Source index for ${part.title}. The HTML course links every entry to an interactive source scene.`);
}

async function writeDeck(part) {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Murali Modular Tech Talk";
  pptx.company = "MAX internals source study";
  pptx.subject = part.subtitle;
  pptx.title = `Llama MAX GPU Masterclass Part ${part.id}: ${part.title}`;
  pptx.lang = "en-US";
  pptx.theme = { headFontFace: "Aptos Display", bodyFontFace: "Aptos", lang: "en-US" };
  const total = 3 + part.scenes.length * 4;
  titleSlide(pptx, part, total);
  routeSlide(pptx, part, total);
  sourceIndexSlide(pptx, part);
  let slideNo = 4;
  for (const scene of part.scenes) {
    for (let build = 1; build <= 4; build += 1) {
      contentSlide(pptx, part, scene, build, slideNo, total);
      slideNo += 1;
    }
  }
  const file = path.join(DECKS, `PART_${String(part.id).padStart(2, "0")}_${part.slug.toUpperCase()}.pptx`);
  await pptx.writeFile({ fileName: file });
  const animation = injectAnimations(file);
  return {
    file: path.relative(OUT, file), slides: total,
    animatedSlides: animation.slides.length,
    clickGroups: animation.slides.reduce((count, entry) => count + entry.groups.length, 0),
    animation,
  };
}

function moduleLinks(activePart) {
  return course.map((part) => `<a class="module-link ${part.id === activePart.id ? "active" : ""}" href="part-${String(part.id).padStart(2, "0")}-${part.slug}.html"><b>${String(part.id).padStart(2, "0")}</b><span>${esc(part.title)}</span></a>`).join("");
}

function courseData(part) {
  return JSON.stringify({
    revision: REV,
    part: { id: part.id, title: part.title, subtitle: part.subtitle, color: part.color },
    scenes: part.scenes,
  }).replaceAll("<", "\\u003c");
}

function partPage(part) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="${esc(part.subtitle)}"><title>Part ${part.id}: ${esc(part.title)}</title><link rel="stylesheet" href="assets/course.css"><script src="assets/course.js" defer></script></head>
<body><div class="course-shell">
  <header class="course-topbar"><a class="course-brand" href="index.html"><span class="brand-index">P${String(part.id).padStart(2, "0")}</span><div><strong>LLAMA / MAX / GPU</strong><span>Source-grounded systems masterclass</span></div></a><div class="part-heading"><small id="scene-title">SOURCE SCENE</small><strong id="part-title">${esc(part.title)}</strong><span id="part-subtitle">${esc(part.subtitle)}</span></div><div class="evidence-legend"><span class="evidence source">SOURCE</span><span class="evidence boundary">BOUNDARY</span><span class="evidence compile-only">COMPILE</span><span class="evidence gpu-lab">GPU-LAB</span><code>${REV.slice(0, 12)}</code></div></header>
  <nav class="module-bar" aria-label="Course modules">${moduleLinks(part)}</nav>
  <main class="workbench">
    <section class="source-pane" aria-label="Repository source"><header class="pane-header"><div><span id="source-evidence" class="evidence source">SOURCE</span><strong id="source-symbol">Source symbol</strong></div><span class="kind-tag">EXACT</span></header><div class="source-path-row"><code id="source-path">path</code><span id="source-range">range</span></div><div id="source-code" class="source-code"></div><footer id="line-explanation" class="line-explanation"></footer></section>
    <section class="visual-pane" aria-label="Interactive system visualization"><header class="visual-toolbar"><div class="diagram-heading"><span id="diagram-kind" class="kind-tag">SYSTEM</span><strong id="diagram-title">Object interaction</strong></div><div class="segmented" role="group" aria-label="Visualization view"><button class="active" data-view="system">System</button><button data-view="math">Math</button><button data-view="evidence">Evidence</button></div></header><div id="visual-canvas" class="visual-canvas"></div><nav id="scene-rail" class="scene-rail" aria-label="Source scenes"></nav></section>
    <aside class="inspector-pane" aria-label="Current state inspector"><header class="pane-header"><div><span id="focus-line" class="kind-tag">L0</span><strong id="step-title">Contract overview</strong></div><span id="step-number" class="step-number">00</span></header><p id="step-summary" class="step-summary"></p><section class="inspector-section"><h2>Object State</h2><div id="object-ledger" class="object-ledger"></div></section><section class="inspector-section"><h2>Measured / Modeled</h2><div id="metric-grid" class="metric-grid"></div></section><section class="inspector-section"><h2>Mathematical Model</h2><code id="formula" class="formula-box"></code></section><section class="inspector-section"><h2>Evidence Boundary</h2><p id="boundary-note" class="boundary-copy"></p></section></aside>
  </main>
  <footer class="transport"><div class="transport-buttons"><button id="previous" title="Previous state" aria-label="Previous state">←</button><button id="play" class="primary" title="Play" aria-label="Play">▶</button><button id="next" title="Next state" aria-label="Next state">→</button><button id="reset" title="Reset scene" aria-label="Reset scene">↺</button></div><div class="build-buttons" aria-label="Build state"><button data-build="0">C</button><button data-build="1">1</button><button data-build="2">2</button><button data-build="3">3</button><button data-build="4">4</button></div><input id="step-slider" type="range" min="0" max="4" value="0" aria-label="Current build"><div class="transport-readout"><strong id="transport-label">Contract overview</strong><span id="transport-count">1/1 · 0/4</span></div></footer>
</div><script id="course-data" type="application/json">${courseData(part)}</script></body></html>`;
}

function indexPage(manifest) {
  const moduleRows = course.map((part) => {
    const deck = manifest.decks.find((entry) => entry.part === part.id);
    return `<a class="module-row" style="--module-color:#${part.color}" href="part-${String(part.id).padStart(2, "0")}-${part.slug}.html"><b>${String(part.id).padStart(2, "0")}</b><div><strong>${esc(part.title)}</strong><span>${esc(part.subtitle)}</span></div><em>${part.units.length} units<br>${part.scenes.length} scenes<br>${deck.slides} slides</em></a>`;
  }).join("");
  const stack = [
    ["Llama request", "Tokens, row offsets, model configuration, weights", course[0]],
    ["MAX Graph", "Symbolic values, GraphOp, output ABI, verification", course[0]],
    ["Graph compiler", "MO -> MAP -> MOGG -> MGP/JIT -> MEF", course[1]],
    ["Mojo kernels", "Registration, tiling, memory movement, matrix instructions", course[2]],
    ["HAL and driver", "Buffers, bundles, queues, CUDA/HIP/Metal plugins", course[3]],
    ["MAX Serve", "Admission, CE/TG batching, chunked prefill, paged KV", course[4]],
  ].map(([title, description, part]) => `<a style="--stack-color:#${part.color}" href="part-${String(part.id).padStart(2, "0")}-${part.slug}.html"><b>${esc(title)}</b><span>${esc(description)}</span></a>`).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Source-grounded MAX graph, compiler, GPU, and serving masterclass."><title>Llama / MAX / GPU Masterclass</title><link rel="stylesheet" href="assets/course.css"></head><body class="index-page"><header class="console-topbar"><div><h1>Llama / MAX / GPU Masterclass</h1><p>Graph construction, compiler lowering, Mojo kernels, device runtime, and MAX Serve in execution order.</p></div><div class="evidence-legend"><span class="evidence source">SOURCE</span><span class="evidence boundary">BOUNDARY</span><span class="evidence gpu-lab">GPU-LAB</span><code>${REV.slice(0, 12)}</code></div></header><main class="console-main"><section class="console-column"><h2>Course Modules</h2><div class="module-list">${moduleRows}</div></section><section class="console-column"><h2>End-to-End Execution Stack</h2><div class="system-stack">${stack}</div></section><aside class="console-column"><h2>Course Inventory</h2><div class="course-stats"><div class="course-stat"><strong>${manifest.totalSlides}</strong><span>PowerPoint slides</span></div><div class="course-stat"><strong>${manifest.totalScenes}</strong><span>interactive source scenes</span></div><div class="course-stat"><strong>${manifest.totalUnits}</strong><span>functions / anchors</span></div><div class="course-stat"><strong>${manifest.sourceFiles}</strong><span>repository files</span></div><div class="course-stat"><strong>${manifest.animatedSlides}</strong><span>slides with click groups</span></div><div class="course-stat"><strong>${manifest.clickGroups}</strong><span>On Click groups</span></div></div><a class="start-link" href="part-01-graph-construction-finalization.html">Start with graph construction</a><a class="source-link" href="source-index.html">Search files and functions</a></aside></main></body></html>`;
}

function sourceIndexPage() {
  const rows = course.flatMap((part) => part.scenes.map((scene) => `<tr data-search="${esc(`${part.title} ${scene.title} ${scene.file} ${scene.anchor} ${scene.evidence}`.toLowerCase())}"><td>${part.id}</td><td><a href="part-${String(part.id).padStart(2, "0")}-${part.slug}.html#scene-${scene.id}/build-4">${esc(scene.title)}</a></td><td><code>${esc(scene.file)}</code></td><td><code>${esc(scene.anchor.replaceAll("\n", " "))}</code></td><td>${esc(scene.evidence)}</td><td><code>L${scene.lines[0].number}-L${scene.lines.at(-1).number}</code></td></tr>`)).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Source Index · Llama MAX GPU Masterclass</title><link rel="stylesheet" href="assets/course.css"></head><body class="source-page"><header class="console-topbar"><div><a href="index.html">← Course console</a><h1>Source and Function Index</h1><p>Search every exact source window used by the HTML course and PowerPoint decks.</p></div><input id="source-search" class="source-search" type="search" placeholder="Filter by file, function, concept, or evidence"></header><main class="source-table-wrap"><table class="source-table"><thead><tr><th>Part</th><th>Scene</th><th>File</th><th>Function / anchor</th><th>Evidence</th><th>Lines</th></tr></thead><tbody>${rows}</tbody></table></main><script>(()=>{const input=document.querySelector('#source-search'),rows=[...document.querySelectorAll('tbody tr')];input.addEventListener('input',()=>{const q=input.value.trim().toLowerCase();rows.forEach(row=>row.hidden=q&&!row.dataset.search.includes(q));});})();</script></body></html>`;
}

function courseMap(manifest) {
  const tree = course.map((part) => {
    const deck = manifest.decks.find((entry) => entry.part === part.id);
    const units = part.units.map((unit, index) => `    ${String(index + 1).padStart(2, "0")}. ${unit.title}\n        File: \`${unit.file}\`\n        Function/anchor: \`${unit.anchor.replaceAll("\n", " ")}\`\n        Role: ${unit.role}\n        Evidence: \`${unit.evidence}\`\n        Math: \`${unit.formula}\``).join("\n");
    return `- Part ${part.id}: ${part.title} (${part.units.length} units, ${part.scenes.length} scenes, ${deck.slides} slides)\n${units}`;
  }).join("\n\n");
  return `# Llama / MAX / GPU Masterclass Course Map\n\nRevision: \`${REV}\`\n\nTotal: **${manifest.totalSlides} PowerPoint slides**, **${manifest.totalScenes} interactive source scenes**, **${manifest.totalUnits} source units**, and **${manifest.sourceFiles} unique source files**.\n\nEach source unit becomes a contract scene and implementation scene. Each scene becomes four progressive PowerPoint builds with three real On Click groups and one five-state HTML laboratory. Evidence is labeled \`SOURCE\`, \`BOUNDARY\`, \`COMPILE-ONLY\`, or \`GPU-LAB\`. Qualcomm is documented only as a boundary; this course does not claim a supported MAX Qualcomm GPU backend.\n\n## Tree\n\n${tree}\n`;
}

function lineByLineAtlas() {
  const sections = course.map((part) => {
    const scenes = part.scenes.map((scene) => {
      const rows = scene.lines.map((entry, index) => `- \`L${entry.number}\` \`${entry.text.trim().replaceAll("`", "\\`")}\`\n  ${scene.explanations[index]}`).join("\n");
      return `### ${scene.title}\n\n- File: \`${scene.file}\`\n- Source range: \`${scene.lines[0].number}-${scene.lines.at(-1).number}\`\n- Function/anchor: \`${scene.anchor.replaceAll("\n", " ")}\`\n- Evidence: \`${scene.evidence}\`\n- Mathematical model: \`${scene.formula}\`\n\n${rows}`;
    }).join("\n\n");
    return `## Part ${part.id}: ${part.title}\n\n${scenes}`;
  }).join("\n\n");
  return `# Line-by-Line Source Atlas\n\nRevision: \`${REV}\`\n\nThis atlas explains every source line displayed in the ${course.reduce((count, part) => count + part.scenes.length, 0)} interactive scenes. The first window captures each callable's contract; the second selects a high-signal implementation block nearby.\n\n${sections}\n`;
}

function readme(manifest) {
  return `# Llama / MAX / GPU Masterclass\n\nA source-driven course covering MAX graph construction, graph compiler transformations, Mojo GPU kernels, driver offload, target backends, and MAX Serve scheduling. The final interface follows the approved V4 synchronized-laboratory model.\n\n- PowerPoint: \`decks/\` (${manifest.totalSlides} slides across six files)\n- Browser course: \`html/index.html\`\n- Searchable source index: \`html/source-index.html\`\n- Source/function map: \`COURSE_MAP.md\`\n- Displayed source line explanations: \`LINE_BY_LINE_SOURCE_ATLAS.md\`\n- Animation inventory: \`animation_manifest.json\`\n- Reproducible generator: \`src/\`\n\n## Build\n\n\`\`\`bash\ncd murali_docs/artifacts/ModularTechTalk/LLAMA_MAX_GPU_MASTERCLASS/src\nnpm install\nnpm run build\nnpm run check\n\`\`\`\n\nRevision used for source excerpts: \`${REV}\`. The decks contain ${manifest.animatedSlides} animated content slides and ${manifest.clickGroups} On Click groups. GPU timing, occupancy, achieved overlap, and bandwidth remain \`GPU-LAB\` items; source inspection is not presented as measured device performance.\n`;
}

function designGuide(manifest) {
  return `# Final Course Design and Validation\n\n## Interaction Grammar\n\nThe HTML and PowerPoint outputs use the same synchronized model:\n\n\`\`\`text\nexact source  <->  object / IR / hardware visualization  <->  state + math + evidence\n                                  |\n                           shared build state\n\`\`\`\n\nEach HTML scene has a contract state plus four progressive builds. Each PowerPoint build slide contains three On Click groups: newly introduced source lines, the corresponding system transition, and the state/math explanation.\n\n## Inventory\n\n- ${manifest.totalSlides} slides across six decks\n- ${manifest.animatedSlides} animated slides\n- ${manifest.clickGroups} click groups\n- ${manifest.totalScenes} HTML source scenes\n- ${manifest.totalUnits} source units\n- ${manifest.sourceFiles} source files\n\n## Evidence Rules\n\n- \`SOURCE\`: exact source proves the displayed contract or transformation.\n- \`BOUNDARY\`: a native compiler, serialized artifact, vendor API, or hardware implementation is below the displayed source.\n- \`COMPILE-ONLY\`: target lowering is inspected without claiming target execution.\n- \`GPU-LAB\`: timing, traffic, overlap, occupancy, and throughput require a future target GPU measurement.\n- Qualcomm is not presented as a supported MAX GPU backend. The source atlas marks the Hexagon/HTP discussion as a boundary.\n\n## Validation\n\nRun \`npm run check\` after generation. It verifies source/anchor resolution, slide counts, notes, transitions, animation timing XML, HTML links, and generated asset syntax. Browser screenshots and PowerPoint rendering checks are performed separately before delivery.\n`;
}

async function main() {
  ensure(DECKS);
  ensure(ASSETS);
  const deckRecords = [];
  const animationManifest = { revision: REV, decks: [] };
  for (const part of course) {
    process.stdout.write(`Generating part ${part.id}: ${part.title}\n`);
    const result = await writeDeck(part);
    deckRecords.push({
      part: part.id, title: part.title, file: result.file, slides: result.slides,
      animatedSlides: result.animatedSlides, clickGroups: result.clickGroups,
    });
    animationManifest.decks.push({
      part: part.id, file: result.file, animatedSlides: result.animatedSlides,
      clickGroups: result.clickGroups,
      slides: result.animation.slides.map((entry) => ({
        slide: entry.slide,
        groups: entry.groups.map((group) => ({ group: group.group, shapes: group.shapeIds.length })),
      })),
    });
    fs.writeFileSync(path.join(HTML, `part-${String(part.id).padStart(2, "0")}-${part.slug}.html`), partPage(part), "utf8");
  }
  const uniqueFiles = new Set(course.flatMap((part) => part.units.map((unit) => unit.file)));
  const manifest = {
    revision: REV, generatedAt: new Date().toISOString(),
    totalSlides: deckRecords.reduce((count, deck) => count + deck.slides, 0),
    totalScenes: course.reduce((count, part) => count + part.scenes.length, 0),
    totalUnits: course.reduce((count, part) => count + part.units.length, 0),
    sourceFiles: uniqueFiles.size,
    animatedSlides: deckRecords.reduce((count, deck) => count + deck.animatedSlides, 0),
    clickGroups: deckRecords.reduce((count, deck) => count + deck.clickGroups, 0),
    decks: deckRecords,
    html: ["html/index.html", "html/source-index.html", ...course.map((part) => `html/part-${String(part.id).padStart(2, "0")}-${part.slug}.html`)],
  };
  fs.copyFileSync(path.join(__dirname, "course.css"), path.join(ASSETS, "course.css"));
  fs.copyFileSync(path.join(__dirname, "course.js"), path.join(ASSETS, "course.js"));
  fs.writeFileSync(path.join(HTML, "index.html"), indexPage(manifest), "utf8");
  fs.writeFileSync(path.join(HTML, "source-index.html"), sourceIndexPage(), "utf8");
  fs.writeFileSync(path.join(OUT, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  fs.writeFileSync(path.join(OUT, "animation_manifest.json"), `${JSON.stringify(animationManifest, null, 2)}\n`, "utf8");
  fs.writeFileSync(path.join(OUT, "COURSE_MAP.md"), courseMap(manifest), "utf8");
  fs.writeFileSync(path.join(OUT, "LINE_BY_LINE_SOURCE_ATLAS.md"), lineByLineAtlas(), "utf8");
  fs.writeFileSync(path.join(OUT, "README.md"), readme(manifest), "utf8");
  fs.writeFileSync(path.join(OUT, "DESIGN_AND_VALIDATION.md"), designGuide(manifest), "utf8");
  process.stdout.write(`Generated ${manifest.totalSlides} slides, ${manifest.animatedSlides} animated slides, and ${manifest.totalScenes} HTML scenes.\n`);
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
