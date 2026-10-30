#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const PptxGenJS = require("pptxgenjs");
const AdmZip = require("adm-zip");
const { parts } = require("./curriculum.cjs");

const ROOT = path.resolve(__dirname, "../../../../..");
const OUT = path.resolve(__dirname, "..");
const DECKS = path.join(OUT, "decks");
const HTML = path.join(OUT, "html");
const ASSETS = path.join(HTML, "assets");
const REV = execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim();
const SHAPE = new PptxGenJS().ShapeType;

const C = {
  ink: "121416", panel: "1B1F22", panel2: "23282C", paper: "F4F1EA",
  muted: "A9B0B4", line: "3B4348", green: "36D17D", teal: "20C997",
  coral: "FF6B4A", yellow: "FFD166", blue: "5AB2FF", violet: "A78BFA",
  red: "FF4D67", white: "FFFFFF", black: "000000",
};

const ensure = (dir) => fs.mkdirSync(dir, { recursive: true });
const esc = (value) => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const clip = (value, n = 95) => value.length > n ? value.slice(0, n - 1) + "..." : value;
const slugify = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

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
      let score = window.reduce((sum, lineText) => {
        const trimmed = lineText.trim();
        let lineScore = trimmed ? 1 : -0.5;
        for (const term of terms) if (lineText.includes(term)) lineScore += 3;
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
  return Array.from({ length: 8 }, (_, i) => ({
    number: start + i + 1,
    text: lines[start + i] === undefined ? "" : lines[start + i],
  }));
}

function explainLine(text) {
  const s = text.trim();
  if (!s) return "A visual separator keeps the source block readable; no runtime action occurs here.";
  if (s.startsWith("#") || s.startsWith("//")) return "The source comment records an invariant, intent, or boundary that constrains the implementation below.";
  if (s.startsWith("@")) return "A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.";
  if (/^(async\s+)?def\s/.test(s) || /^[\w:<>,*&\s]+\([^;]*\)\s*\{?$/.test(s)) return "This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.";
  if (/^(class|struct)\s/.test(s)) return "This type creates an ownership and behavior boundary for the state used by later calls.";
  if (/^(if|elif|else|switch|case|comptime if)\b/.test(s)) return "This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.";
  if (/^(for|while|comptime for)\b/.test(s)) return "This loop expands repeated work while preserving the surrounding ordering and dependency rules.";
  if (s.includes("addPass") || s.includes("buildStage")) return "This line fixes transformation order; downstream passes consume the invariants produced here.";
  if (s.includes("enqueue_")) return "This enqueues asynchronous device work; host completion is not implied until an explicit synchronization or dependency requires it.";
  if (s.includes("verify")) return "This validates the current IR or object contract and surfaces diagnostics close to the originating source operation.";
  if (s.startsWith("return")) return "The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.";
  if (s.includes("=")) return "This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.";
  return "This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.";
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
      });
    }
  });
  return scenes;
}

const course = parts.map((p) => ({ ...p, scenes: buildScenes(p) }));

function addText(slide, text, x, y, w, h, opt = {}) {
  slide.addText(text, {
    x, y, w, h, fontFace: opt.mono ? "Aptos Mono" : "Aptos",
    fontSize: opt.size || 13, color: opt.color || C.paper,
    bold: !!opt.bold, margin: opt.margin === undefined ? 0 : opt.margin,
    valign: opt.valign || "mid", align: opt.align || "left",
    fit: "shrink", breakLine: false, ...opt,
  });
}

function rect(slide, x, y, w, h, fill, line = fill, radius = false) {
  slide.addShape(radius ? SHAPE.roundRect : SHAPE.rect, {
    x, y, w, h, fill: { color: fill }, line: { color: line, width: 1 },
    radius: radius ? 0.05 : undefined,
  });
}

function line(slide, x, y, w, h, color = C.line, arrow = false, width = 1.5) {
  slide.addShape(SHAPE.line, {
    x, y, w, h,
    line: { color, width, endArrowType: arrow ? "triangle" : "none" },
  });
}

function circle(slide, x, y, d, fill, label, opt = {}) {
  slide.addShape(SHAPE.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: opt.line || fill, width: 1.2 } });
  addText(slide, label, x + 0.04, y + 0.04, d - 0.08, d - 0.08, { size: opt.size || 10, bold: true, color: opt.color || C.ink, align: "center" });
}

function notes(slide, text) {
  if (typeof slide.addNotes === "function") slide.addNotes(text);
}

function header(slide, part, scene, build) {
  slide.background = { color: C.ink };
  rect(slide, 0, 0, 0.14, 7.5, part.color);
  addText(slide, `PART ${part.id}  /  ${scene.diagram.toUpperCase()}`, 0.42, 0.22, 4.4, 0.26, { size: 9.5, color: part.color, bold: true });
  addText(slide, scene.title, 0.42, 0.52, 9.9, 0.52, { size: 25, bold: true });
  addText(slide, clip(scene.role, 150), 0.44, 1.07, 11.6, 0.38, { size: 10.5, color: C.muted });
  rect(slide, 11.65, 0.27, 1.15, 0.34, C.panel2, part.color, true);
  addText(slide, `BUILD ${build}/4`, 11.65, 0.27, 1.15, 0.34, { size: 8.5, color: part.color, bold: true, align: "center" });
}

function footer(slide, part, scene, slideNo, total) {
  addText(slide, scene.evidence, 0.43, 7.15, 2.6, 0.16, { size: 7.5, color: scene.evidence.includes("BOUNDARY") ? C.yellow : part.color, bold: true });
  addText(slide, `${REV.slice(0, 10)}  |  ${scene.file}:${scene.lines[0].number}`, 3.1, 7.15, 7.1, 0.16, { size: 7.2, color: C.muted, mono: true, align: "center" });
  addText(slide, `${slideNo} / ${total}`, 11.55, 7.15, 1.2, 0.16, { size: 7.5, color: C.muted, align: "right" });
}

function codePanel(slide, scene, build, part) {
  rect(slide, 0.43, 1.62, 7.36, 4.78, C.panel, C.line, true);
  rect(slide, 0.43, 1.62, 7.36, 0.42, C.panel2, C.line);
  addText(slide, clip(scene.file, 78), 0.66, 1.7, 6.85, 0.2, { size: 8.2, color: part.color, mono: true, bold: true });
  const visible = build * 2;
  scene.lines.forEach((entry, i) => {
    const y = 2.13 + i * 0.5;
    const shown = i < visible;
    const newest = shown && i >= visible - 2;
    if (newest) rect(slide, 0.51, y - 0.03, 7.18, 0.43, "293238", part.color);
    else if (shown) rect(slide, 0.51, y - 0.03, 7.18, 0.43, i % 2 ? "1D2225" : C.panel);
    else rect(slide, 0.51, y - 0.03, 7.18, 0.43, "171A1C");
    addText(slide, String(entry.number), 0.63, y, 0.42, 0.24, { size: 8.5, color: shown ? C.muted : C.line, mono: true, align: "right" });
    addText(slide, shown ? clip(entry.text, 94) : "", 1.18, y, 6.3, 0.25, { size: 8.8, color: newest ? C.white : C.paper, mono: true });
  });
  const focus = scene.lines[Math.min(visible - 1, scene.lines.length - 1)];
  rect(slide, 0.43, 6.54, 7.36, 0.43, C.panel2, part.color, true);
  addText(slide, `L${focus.number}: ${clip(explainLine(focus.text), 145)}`, 0.62, 6.61, 6.95, 0.23, { size: 7.9, color: C.paper });
}

function conceptLabel(slide, text, x, y, w, h, part, active = true, opt = {}) {
  rect(slide, x, y, w, h, active ? C.panel2 : "191C1E", active ? part.color : C.line, true);
  addText(slide, text, x + 0.08, y + 0.05, w - 0.16, h - 0.1, { size: opt.size || 9, color: active ? C.paper : C.line, bold: active, align: opt.align || "center" });
}

function diagramFlow(slide, scene, build, part, vertical = false) {
  const visible = build;
  scene.concepts.forEach((c, i) => {
    const x = vertical ? 8.52 : 8.08 + (i % 2) * 2.32;
    const y = vertical ? 1.9 + i * 1.02 : 2.0 + Math.floor(i / 2) * 1.65;
    conceptLabel(slide, c, x, y, vertical ? 3.7 : 2.05, 0.68, part, i < visible);
    if (i > 0 && i < visible) {
      if (vertical) line(slide, 10.37, y - 0.34, 0, 0.32, part.color, true, 2);
      else if (i % 2 === 1) line(slide, x - 0.26, y + 0.34, 0.22, 0, part.color, true, 2);
    }
  });
}

function diagramIR(slide, scene, build, part) {
  addText(slide, "BEFORE", 8.04, 1.77, 1.2, 0.22, { size: 8, color: C.muted, bold: true });
  addText(slide, "AFTER", 10.72, 1.77, 1.2, 0.22, { size: 8, color: part.color, bold: true });
  conceptLabel(slide, scene.concepts[0], 8.02, 2.13, 1.75, 0.72, part, build >= 1);
  conceptLabel(slide, scene.concepts[1], 8.02, 3.17, 1.75, 0.72, part, build >= 2);
  line(slide, 9.92, 2.95, 0.58, 0, part.color, build >= 3, 2.5);
  conceptLabel(slide, scene.concepts[2], 10.67, 2.13, 1.78, 0.72, part, build >= 3);
  conceptLabel(slide, scene.concepts[3], 10.67, 3.17, 1.78, 0.72, part, build >= 4);
  if (build >= 3) {
    addText(slide, "PASS", 9.79, 2.47, 0.82, 0.28, { size: 8, color: part.color, bold: true, align: "center" });
  }
}

function diagramMemory(slide, scene, build, part) {
  const labels = scene.concepts;
  addText(slide, "HOST / OWNER", 8.02, 2.02, 1.35, 0.2, { size: 7.5, color: C.muted, bold: true });
  addText(slide, "DEVICE / NEXT STATE", 10.55, 2.02, 1.7, 0.2, { size: 7.5, color: part.color, bold: true });
  for (let i = 0; i < 4; i += 1) {
    const y = 2.36 + i * 0.75;
    rect(slide, 8.04, y, 1.62, 0.48, i < build ? "2B3236" : "191C1E", i < build ? C.muted : C.line, true);
    rect(slide, 10.74, y, 1.62, 0.48, i < build ? C.panel2 : "191C1E", i < build ? part.color : C.line, true);
    addText(slide, clip(labels[i], 22), 8.12, y + 0.07, 1.46, 0.3, { size: 7.5, color: i < build ? C.paper : C.line, align: "center" });
    addText(slide, i < build ? `next: ${clip(labels[i], 16)}` : "", 10.82, y + 0.07, 1.46, 0.3, { size: 7.2, color: i < build ? C.paper : C.line, align: "center" });
    if (i < build) line(slide, 9.72, y + 0.24, 0.92, 0, i % 2 ? C.yellow : part.color, true, 2);
  }
}

function diagramTile(slide, scene, build, part) {
  const x0 = 8.15, y0 = 1.98, cell = 0.52;
  for (let r = 0; r < 6; r += 1) {
    for (let c = 0; c < 7; c += 1) {
      const stage = Math.floor(c / 2) + 1;
      rect(slide, x0 + c * cell, y0 + r * cell, cell - 0.04, cell - 0.04, stage <= build ? (r < 3 ? part.color : C.panel2) : "191C1E", stage <= build ? C.line : "2A2F32");
    }
  }
  for (let i = 0; i < 4; i += 1) addText(slide, scene.concepts[i], 8.1 + (i % 2) * 2.1, 5.3 + Math.floor(i / 2) * 0.45, 1.9, 0.28, { size: 7.5, color: i < build ? part.color : C.line, bold: i < build, align: "center" });
}

function diagramTimeline(slide, scene, build, part) {
  const lanes = ["CPU", "COPY", "GPU", "RETIRE"];
  lanes.forEach((laneName, i) => {
    const y = 2.0 + i * 0.85;
    addText(slide, laneName, 8.02, y, 0.72, 0.28, { size: 7.5, color: C.muted, bold: true });
    line(slide, 8.82, y + 0.16, 3.62, 0, C.line, false, 1);
    if (i < build) {
      const x = 8.9 + i * 0.46;
      rect(slide, x, y - 0.02, 1.58 + (3 - i) * 0.18, 0.36, i % 2 ? part.color : C.panel2, i % 2 ? part.color : C.muted, true);
      addText(slide, clip(scene.concepts[i], 25), x + 0.08, y + 0.03, 1.65, 0.23, { size: 7.2, color: i % 2 ? C.ink : C.paper, bold: true, align: "center" });
    }
  });
}

function diagramRoofline(slide, scene, build, part) {
  const x = 8.25, y = 5.02, w = 3.75, h = 2.75;
  line(slide, x, y, w, 0, C.muted, true, 1.5);
  line(slide, x, y, 0, -h, C.muted, true, 1.5);
  if (build >= 1) line(slide, x + 0.12, y - 0.12, 2.05, -1.92, part.color, false, 3);
  if (build >= 2) line(slide, x + 2.17, y - 2.04, 1.46, 0, C.yellow, false, 3);
  if (build >= 3) circle(slide, x + 1.61, y - 1.71, 0.28, C.red, "", { line: C.white });
  if (build >= 4) circle(slide, x + 2.55, y - 2.18, 0.28, C.green, "", { line: C.white });
  addText(slide, "Arithmetic intensity ->", 8.73, 5.18, 2.9, 0.22, { size: 7.5, color: C.muted, align: "center" });
  addText(slide, "Performance", 7.9, 2.3, 0.3, 1.3, { size: 7.5, color: C.muted, rotate: 270, align: "center" });
}

function diagramBackend(slide, scene, build, part) {
  conceptLabel(slide, scene.concepts[0], 9.25, 1.83, 2.0, 0.58, part, build >= 1);
  for (let i = 1; i < 4; i += 1) {
    const y = 2.74 + (i - 1) * 0.92;
    if (i < build) line(slide, 10.23, 2.4, (i - 2) * 0.58, y - 2.42, part.color, true, 1.8);
    conceptLabel(slide, scene.concepts[i], 8.12 + (i % 2) * 2.25, y, 2.02, 0.58, part, i < build);
  }
}

function diagramState(slide, scene, build, part) {
  scene.concepts.forEach((c, i) => {
    const x = 8.08 + (i % 2) * 2.28;
    const y = 2.0 + Math.floor(i / 2) * 1.55;
    circle(slide, x + 0.48, y, 0.82, i < build ? part.color : C.panel2, String(i + 1), { color: i < build ? C.ink : C.line, line: i < build ? part.color : C.line });
    addText(slide, c, x, y + 0.91, 1.78, 0.35, { size: 7.7, color: i < build ? C.paper : C.line, bold: i < build, align: "center" });
  });
  if (build >= 2) line(slide, 9.42, 2.41, 0.9, 0, part.color, true, 1.8);
  if (build >= 3) line(slide, 10.77, 2.85, 0, 0.68, part.color, true, 1.8);
  if (build >= 4) line(slide, 10.32, 3.96, -0.9, 0, part.color, true, 1.8);
}

function diagramFor(slide, scene, build, part) {
  rect(slide, 7.98, 1.62, 4.82, 4.78, C.panel, C.line, true);
  const type = scene.diagram;
  if (!["memory", "buffers", "cache"].includes(type)) {
    addText(slide, scene.diagram.toUpperCase(), 8.18, 1.72, 2.0, 0.2, { size: 8, color: part.color, bold: true });
  }
  if (["pipeline", "sequence", "graph"].includes(type)) diagramFlow(slide, scene, build, part, type === "sequence");
  else if (type === "ir") diagramIR(slide, scene, build, part);
  else if (["memory", "buffers", "cache"].includes(type)) diagramMemory(slide, scene, build, part);
  else if (type === "tile") diagramTile(slide, scene, build, part);
  else if (["timeline", "schedule", "queue"].includes(type)) diagramTimeline(slide, scene, build, part);
  else if (type === "roofline") diagramRoofline(slide, scene, build, part);
  else if (["backend", "dispatch", "boundary"].includes(type)) diagramBackend(slide, scene, build, part);
  else diagramState(slide, scene, build, part);
  rect(slide, 7.98, 6.54, 4.82, 0.43, C.panel2, part.color, true);
  addText(slide, scene.formula, 8.13, 6.61, 4.52, 0.23, { size: 8.2, color: C.paper, mono: true, align: "center" });
}

function titleSlide(pptx, part, slideNo, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.ink };
  rect(slide, 0.7, 0.6, 0.17, 5.95, part.color);
  addText(slide, `LLAMA / MAX / GPU MASTERCLASS  |  PART ${part.id}`, 1.12, 0.75, 7.7, 0.32, { size: 12, color: part.color, bold: true });
  addText(slide, part.title, 1.08, 1.32, 8.2, 1.3, { size: 37, bold: true, valign: "top" });
  addText(slide, part.subtitle, 1.12, 2.93, 7.35, 0.78, { size: 17, color: C.muted, valign: "top" });
  const stages = part.units.slice(0, 4);
  stages.forEach((u, i) => {
    conceptLabel(slide, u.title, 9.28, 1.0 + i * 1.18, 3.25, 0.72, part, true, { size: 10 });
    if (i < stages.length - 1) line(slide, 10.9, 1.73 + i * 1.18, 0, 0.42, part.color, true, 2);
  });
  addText(slide, `${part.scenes.length} source scenes  /  4 progressive builds each`, 1.12, 5.76, 6.4, 0.3, { size: 12, color: C.yellow, bold: true });
  addText(slide, `SOURCE REV ${REV.slice(0, 12)}`, 1.12, 6.24, 4.2, 0.24, { size: 9, color: C.muted, mono: true });
  footer(slide, part, { evidence: "SOURCE + BOUNDARY", file: "course manifest", lines: [{ number: 1 }] }, slideNo, total);
  notes(slide, `${part.title}\n\n${part.subtitle}\n\nThis deck contains ${part.scenes.length} source scenes. Each scene uses four progressive slides so PowerPoint transitions reveal source lines and object interactions in order.`);
}

function routeSlide(pptx, part, slideNo, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.ink };
  addText(slide, `PART ${part.id} SOURCE ROUTE`, 0.65, 0.45, 4.2, 0.28, { size: 10, color: part.color, bold: true });
  addText(slide, "Objects and functions in execution order", 0.65, 0.85, 8.8, 0.58, { size: 27, bold: true });
  part.units.forEach((u, i) => {
    const col = i < 6 ? 0 : 1;
    const row = i < 6 ? i : i - 6;
    const x = 0.72 + col * 6.18;
    const y = 1.72 + row * 0.82;
    circle(slide, x, y, 0.45, part.color, String(i + 1), { size: 8.5 });
    addText(slide, u.title, x + 0.62, y - 0.01, 4.95, 0.24, { size: 10.2, bold: true });
    addText(slide, clip(u.anchor.replaceAll("\n", " "), 64), x + 0.62, y + 0.27, 4.95, 0.2, { size: 7.4, color: C.muted, mono: true });
    if (row < 5 && (i < 5 || i >= 6)) line(slide, x + 0.22, y + 0.47, 0, 0.31, C.line, true, 1.3);
  });
  footer(slide, part, { evidence: "SOURCE", file: "curriculum.cjs", lines: [{ number: 1 }] }, slideNo, total);
  notes(slide, `Source route for ${part.title}. The numbered functions form the teaching order. Each function is split into two eight-line source scenes.`);
}

function sourceIndexSlide(pptx, part, slideNo, total) {
  const slide = pptx.addSlide();
  slide.background = { color: C.ink };
  addText(slide, `PART ${part.id} SOURCE INDEX`, 0.65, 0.45, 4.2, 0.28, { size: 10, color: part.color, bold: true });
  addText(slide, "Files, anchors, and evidence", 0.65, 0.84, 7.2, 0.54, { size: 27, bold: true });
  part.units.forEach((u, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.72 + col * 6.18, y = 1.64 + row * 0.86;
    rect(slide, x, y, 5.78, 0.66, C.panel, C.line, true);
    addText(slide, `${String(i + 1).padStart(2, "0")}  ${u.title}`, x + 0.15, y + 0.07, 5.38, 0.22, { size: 9.4, color: part.color, bold: true });
    addText(slide, clip(`${u.file} :: ${u.anchor.replaceAll("\n", " ")}`, 85), x + 0.15, y + 0.35, 5.38, 0.18, { size: 6.9, color: C.muted, mono: true });
  });
  footer(slide, part, { evidence: "SOURCE", file: "source index", lines: [{ number: 1 }] }, slideNo, total);
  notes(slide, `File and function index for ${part.title}. Use the browser source index for search and direct navigation to each animated scene.`);
}

function sceneSlides(pptx, part, scene, startNo, total) {
  for (let build = 1; build <= 4; build += 1) {
    const slide = pptx.addSlide();
    header(slide, part, scene, build);
    codePanel(slide, scene, build, part);
    diagramFor(slide, scene, build, part);
    footer(slide, part, scene, startNo + build - 1, total);
    const visible = scene.lines.slice(0, build * 2);
    const current = visible[visible.length - 1];
    notes(slide, [
      scene.title,
      "",
      scene.role,
      "",
      `Evidence: ${scene.evidence}`,
      `Source: ${scene.file}:${scene.lines[0].number}-${scene.lines[scene.lines.length - 1].number}`,
      `Anchor: ${scene.anchor.replaceAll("\n", " ")}`,
      `Build ${build}/4 focuses through line ${current.number}.`,
      "",
      ...visible.map((entry) => `L${entry.number}: ${entry.text.trim()}\n  ${explainLine(entry.text)}`),
      "",
      `Mathematical model: ${scene.formula}`,
      "",
      "Performance durations and achieved overlap are not claimed unless separately measured on a target GPU; GPU-LAB marks those experiments.",
    ].join("\n"));
  }
}

function injectTransitions(file) {
  const zip = new AdmZip(file);
  const entries = zip.getEntries().filter((entry) => /^ppt\/slides\/slide\d+\.xml$/.test(entry.entryName));
  entries.forEach((entry, i) => {
    let xml = entry.getData().toString("utf8");
    if (xml.includes("<p:transition")) return;
    const effect = i % 3 === 0 ? "fade" : i % 3 === 1 ? "wipe dir=\"r\"" : "push dir=\"l\"";
    const [tag, attrs = ""] = effect.split(" ", 2);
    const transition = `<p:transition spd="fast" advClick="1"><p:${tag}${attrs ? " " + attrs : ""}/></p:transition>`;
    if (xml.includes("<p:clrMapOvr")) xml = xml.replace("<p:clrMapOvr", transition + "<p:clrMapOvr");
    else xml = xml.replace("</p:sld>", transition + "</p:sld>");
    zip.updateFile(entry.entryName, Buffer.from(xml, "utf8"));
  });
  zip.writeZip(file);
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
  titleSlide(pptx, part, 1, total);
  routeSlide(pptx, part, 2, total);
  let slideNo = 3;
  part.scenes.forEach((scene) => {
    sceneSlides(pptx, part, scene, slideNo, total);
    slideNo += 4;
  });
  sourceIndexSlide(pptx, part, total, total);
  const file = path.join(DECKS, `PART_${String(part.id).padStart(2, "0")}_${part.slug.toUpperCase()}.pptx`);
  await pptx.writeFile({ fileName: file });
  injectTransitions(file);
  return { file: path.relative(OUT, file), slides: total };
}

function htmlDiagram(scene) {
  const nodes = scene.concepts.map((concept, i) =>
    `<button class="d-node" data-step="${i + 1}" data-detail="${esc(concept)}" title="Inspect ${esc(concept)}"><b>${i + 1}</b><span>${esc(concept)}</span></button>`,
  ).join("");
  if (scene.diagram === "roofline") {
    return `<div class="diagram roofline" data-kind="roofline"><i class="axis-x"></i><i class="axis-y"></i><i class="memory-roof" data-step="1"></i><i class="compute-roof" data-step="2"></i><i class="point p1" data-step="3"></i><i class="point p2" data-step="4"></i><small>Arithmetic intensity</small></div>`;
  }
  if (scene.diagram === "tile") {
    return `<div class="diagram tile" data-kind="tile">${Array.from({ length: 48 }, (_, i) => `<i data-step="${Math.min(4, Math.floor((i % 8) / 2) + 1)}"></i>`).join("")}<div class="tile-labels">${nodes}</div></div>`;
  }
  if (["memory", "buffers", "cache"].includes(scene.diagram)) {
    return `<div class="diagram memory" data-kind="${scene.diagram}"><header><span>HOST / OWNER</span><span>DEVICE / CACHE</span></header>${scene.concepts.map((c, i) => `<div class="mem-row" data-step="${i + 1}"><i>${esc(c)}</i><em></em><b>${esc(c)}</b></div>`).join("")}</div>`;
  }
  if (["timeline", "schedule", "queue"].includes(scene.diagram)) {
    return `<div class="diagram timeline" data-kind="${scene.diagram}">${scene.concepts.map((c, i) => `<div class="lane" data-step="${i + 1}"><small>${["CPU", "COPY", "GPU", "RETIRE"][i]}</small><i style="--offset:${i};--span:${4 - i}">${esc(c)}</i></div>`).join("")}</div>`;
  }
  return `<div class="diagram nodes kind-${esc(scene.diagram)}" data-kind="${esc(scene.diagram)}">${nodes}</div>`;
}

function htmlScene(scene, part) {
  const code = scene.lines.map((entry, i) => {
    const step = Math.floor(i / 2) + 1;
    return `<button class="code-line" data-step="${step}" data-line="${entry.number}" data-detail="${esc(explainLine(entry.text))}" title="Inspect source line ${entry.number}"><span>${entry.number}</span><code>${esc(entry.text || " ")}</code></button>`;
  }).join("");
  return `<article class="scene" id="scene-${scene.id}" data-scene="${scene.number - 1}" data-steps="4" style="--accent:#${part.color}">
    <header class="scene-head"><div><b>PART ${part.id} / ${esc(scene.diagram.toUpperCase())}</b><h1>${esc(scene.title)}</h1><p>${esc(scene.role)}</p></div><span class="evidence">${esc(scene.evidence)}</span></header>
    <div class="scene-body">
      <section class="source-pane"><div class="source-title"><a href="../../../../../${esc(scene.file)}" title="Open repository source">${esc(scene.file)}</a><span>${scene.lines[0].number}-${scene.lines[7].number}</span></div>${code}</section>
      <section class="visual-pane">${htmlDiagram(scene)}<div class="formula"><span>MATH</span><code>${esc(scene.formula)}</code></div></section>
    </div>
    <footer><code>${REV.slice(0, 12)}</code><span>${scene.number} / ${part.scenes.length}</span></footer>
  </article>`;
}

function partPage(part) {
  const nav = course.map((p) => `<a class="part-link ${p.id === part.id ? "active" : ""}" href="part-${String(p.id).padStart(2, "0")}-${p.slug}.html" title="Part ${p.id}: ${esc(p.title)}">${p.id}</a>`).join("");
  const dots = part.scenes.map((s, i) => `<button class="scene-dot" data-target="${i}" title="${esc(s.title)}"></button>`).join("");
  const scenes = part.scenes.map((s) => htmlScene(s, part)).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Part ${part.id}: ${esc(part.title)}</title><link rel="stylesheet" href="assets/course.css"></head>
  <body class="course-page"><div class="course-shell">
    <nav class="topbar"><a class="brand" href="index.html">LLAMA / MAX / GPU</a><div class="part-nav">${nav}</div><a class="source-link" href="source-index.html">SOURCE INDEX</a></nav>
    <aside class="rail">${dots}</aside><main class="viewport">${scenes}</main>
    <div class="controls"><span class="build-label"></span><button class="prev" title="Previous build" aria-label="Previous build">&#9664;</button><button class="play" title="Auto play" aria-label="Auto play">&#9654;</button><button class="next" title="Next build" aria-label="Next build">&#9654;&#9654;</button><button class="reset" title="Restart scene" aria-label="Restart scene">&#8634;</button><button class="full" title="Full screen" aria-label="Full screen">&#9974;</button><span class="scene-count"></span></div>
    <aside class="inspector"><button class="close" title="Close inspector" aria-label="Close inspector">&#10005;</button><b>OBJECT / LINE</b><h2></h2><h3>Source</h3><code class="inspect-source"></code><h3>Explanation</h3><p class="inspect-detail"></p><h3>Invariant</h3><code class="inspect-formula"></code></aside>
  </div><script src="assets/course.js"></script></body></html>`;
}

function indexPage(manifest) {
  const rows = course.map((p) => {
    const deck = manifest.decks.find((d) => d.part === p.id);
    return `<a class="part-row" href="part-${String(p.id).padStart(2, "0")}-${p.slug}.html" style="--accent:#${p.color}"><b>0${p.id}</b><span><strong>${esc(p.title)}</strong><small>${esc(p.subtitle)}</small></span><em>${p.scenes.length} scenes<br>${deck.slides} slides</em></a>`;
  }).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Llama MAX GPU Masterclass</title><link rel="stylesheet" href="assets/course.css"></head><body class="index-page">
    <header class="index-hero"><div><b>SOURCE-DRIVEN TECHNICAL COURSE</b><h1>Llama / MAX / GPU<br>Masterclass</h1><p>Graph construction, compiler transformations, Mojo kernels, device offload, backend targets, and MAX Serve scheduling traced through repository source.</p></div><div class="stack-viz"><i>MAX SERVE</i><i>MAX GRAPH</i><i>GRAPH COMPILER</i><i>MOJO + HAL</i><i>GPU / DEVICE</i></div></header>
    <main class="index-main"><section class="stats"><span><b>${manifest.totalSlides}</b> PowerPoint slides</span><span><b>${manifest.totalScenes}</b> animated source scenes</span><span><b>${manifest.sourceFiles}</b> source files</span><span><b>6</b> navigable modules</span></section><section class="part-list">${rows}</section></main>
    <footer class="index-footer"><a href="source-index.html">Search source and functions</a><code>REV ${REV.slice(0, 12)}</code></footer></body></html>`;
}

function sourceIndexPage() {
  const rows = course.flatMap((p) => p.units.map((u, i) => {
    const firstScene = p.scenes[i * 2];
    return `<tr data-search="${esc(`${p.title} ${u.title} ${u.file} ${u.anchor} ${u.role}`.toLowerCase())}"><td>${p.id}</td><td><a href="part-${String(p.id).padStart(2, "0")}-${p.slug}.html#scene-${firstScene.id}">${esc(u.title)}</a></td><td><code>${esc(u.file)}</code></td><td><code>${esc(u.anchor.replaceAll("\n", " "))}</code></td><td>${esc(u.evidence)}</td></tr>`;
  })).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Source Index</title><link rel="stylesheet" href="assets/course.css"></head><body class="source-page"><header class="source-head"><a href="index.html">LLAMA / MAX / GPU</a><h1>Source and function index</h1><input id="source-search" type="search" placeholder="Filter by file, function, object, or concept" aria-label="Filter source index"></header><main><table><thead><tr><th>Part</th><th>Topic</th><th>File</th><th>Function / anchor</th><th>Evidence</th></tr></thead><tbody>${rows}</tbody></table></main><script>const q=document.querySelector('#source-search');q.addEventListener('input',()=>{const v=q.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(v))});</script></body></html>`;
}

const COURSE_CSS = String.raw`
:root{--ink:#121416;--panel:#1b1f22;--panel2:#23282c;--paper:#f4f1ea;--muted:#a9b0b4;--line:#3b4348;--green:#36d17d;--teal:#20c997;--coral:#ff6b4a;--yellow:#ffd166;--blue:#5ab2ff;--violet:#a78bfa;--red:#ff4d67}*{box-sizing:border-box;letter-spacing:0}html,body{margin:0;min-height:100%;background:var(--ink);color:var(--paper);font-family:Arial,Helvetica,sans-serif}button,input{font:inherit}.course-page{height:100vh;overflow:hidden}.course-shell{height:100%;display:grid;grid-template-columns:32px 1fr;grid-template-rows:48px 1fr 58px}.topbar{grid-column:1/3;display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:0 22px;border-bottom:1px solid #292f32;background:#15181a}.brand,.source-link{color:var(--paper);text-decoration:none;font-size:10px;font-weight:800}.source-link{justify-self:end;color:var(--muted)}.part-nav{display:flex;gap:6px}.part-link{display:grid;place-items:center;width:25px;height:25px;border:1px solid var(--line);color:var(--muted);text-decoration:none;font-size:10px}.part-link.active{border-color:var(--accent,#36d17d);color:var(--paper);background:#242a2e}.rail{grid-row:2;border-right:1px solid #292f32;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:5px}.scene-dot{width:7px;height:7px;padding:0;border:1px solid #667076;background:transparent;cursor:pointer}.scene-dot.active{background:var(--accent,#36d17d);border-color:var(--accent,#36d17d)}.viewport{grid-row:2;position:relative;overflow:hidden}.scene{position:absolute;inset:0;padding:22px 34px 14px;display:grid;grid-template-rows:auto 1fr auto;opacity:0;transform:translateY(12px);pointer-events:none;transition:opacity .34s,transform .34s}.scene.active{opacity:1;transform:none;pointer-events:auto}.scene-head{display:flex;justify-content:space-between;gap:24px;align-items:start}.scene-head b{display:block;color:var(--accent);font-size:10px;margin-bottom:7px}.scene-head h1{font-size:28px;line-height:1.05;margin:0 0 7px}.scene-head p{max-width:980px;margin:0;color:var(--muted);font-size:12px;line-height:1.4}.evidence{border:1px solid var(--accent);padding:7px 9px;color:var(--accent);font-size:8px;font-weight:800;white-space:nowrap}.scene-body{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(330px,1fr);gap:15px;min-height:0;padding:16px 0 10px}.source-pane,.visual-pane{min-height:0;border:1px solid var(--line);background:var(--panel);overflow:hidden}.source-pane{display:grid;grid-template-rows:38px repeat(8,1fr)}.source-title{display:flex;align-items:center;justify-content:space-between;padding:0 13px;background:var(--panel2);font:9px monospace;color:var(--muted)}.source-title a{color:var(--accent);text-decoration:none}.code-line{display:grid;grid-template-columns:50px 1fr;text-align:left;align-items:center;width:100%;min-height:0;padding:0;border:0;border-top:1px solid #252b2e;background:#1a1e20;color:var(--paper);opacity:.16;transform:translateX(-8px);transition:opacity .35s,transform .35s,background .25s;cursor:pointer}.code-line.revealed{opacity:1;transform:none}.code-line.current{background:#293238;box-shadow:inset 4px 0 var(--accent)}.code-line span{padding-right:12px;text-align:right;color:var(--muted);font:9px monospace}.code-line code{padding-right:12px;white-space:pre;overflow:hidden;text-overflow:ellipsis;font:10px/1.2 monospace}.visual-pane{display:grid;grid-template-rows:1fr 46px;padding:13px}.formula{display:grid;grid-template-columns:54px 1fr;align-items:center;border-top:1px solid var(--line)}.formula span{color:var(--accent);font-size:8px;font-weight:800}.formula code{font:9px monospace;color:var(--paper);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.diagram{position:relative;min-height:0}.diagram.nodes{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-content:center;padding:18px}.d-node{position:relative;min-height:82px;border:1px solid var(--line);background:#191d1f;color:var(--line);opacity:.25;transform:scale(.96);transition:opacity .4s,transform .4s,color .4s,border-color .4s,background .4s;cursor:pointer}.d-node.revealed{opacity:1;transform:none;color:var(--paper);border-color:var(--accent);background:var(--panel2)}.d-node b{position:absolute;left:8px;top:7px;color:var(--accent);font-size:8px}.d-node span{display:block;padding:22px 10px 10px;font-size:11px;font-weight:700}.kind-sequence{grid-template-columns:1fr!important;padding-left:76px!important}.kind-sequence .d-node{min-height:52px}.kind-pipeline .d-node.revealed:after,.kind-graph .d-node.revealed:after,.kind-dispatch .d-node.revealed:after{content:'';position:absolute;right:-19px;top:50%;width:18px;height:2px;background:var(--accent);animation:pulse 1.25s infinite}.kind-boundary .d-node{border-style:dashed}.memory{display:grid;grid-template-rows:28px repeat(4,1fr);gap:8px;padding:16px}.memory header{display:flex;justify-content:space-between;color:var(--muted);font-size:8px;font-weight:800}.mem-row{display:grid;grid-template-columns:1fr 46px 1fr;align-items:center;opacity:.15;transition:opacity .4s}.mem-row.revealed{opacity:1}.mem-row i,.mem-row b{display:grid;place-items:center;min-height:42px;border:1px solid var(--line);font-size:9px;font-style:normal}.mem-row b{border-color:var(--accent);background:var(--panel2)}.mem-row em{height:2px;background:var(--accent);position:relative;animation:pulse 1.2s infinite}.mem-row em:after{content:'';position:absolute;right:-1px;top:-3px;border-left:6px solid var(--accent);border-top:4px solid transparent;border-bottom:4px solid transparent}.timeline{display:grid;align-content:center;gap:17px;padding:20px}.lane{display:grid;grid-template-columns:48px 1fr;align-items:center;opacity:.15;transition:opacity .4s}.lane.revealed{opacity:1}.lane small{color:var(--muted);font-size:8px;font-weight:800}.lane i{display:block;margin-left:calc(var(--offset)*9%);width:calc(28% + var(--span)*8%);padding:10px;background:var(--panel2);border:1px solid var(--accent);font-size:9px;font-style:normal;text-align:center}.tile{display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(6,1fr);gap:4px;padding:15px 18px 86px}.tile>i{border:1px solid #30373b;background:#171a1c;opacity:.2;transition:opacity .4s,background .4s}.tile>i.revealed{opacity:1;background:var(--accent)}.tile-labels{position:absolute;left:18px;right:18px;bottom:8px;display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.tile-labels .d-node{min-height:46px}.roofline{margin:28px}.roofline .axis-x,.roofline .axis-y{position:absolute;background:var(--muted)}.roofline .axis-x{left:14%;right:4%;bottom:15%;height:2px}.roofline .axis-y{left:14%;top:6%;bottom:15%;width:2px}.roofline .memory-roof,.roofline .compute-roof{position:absolute;height:3px;background:var(--accent);transform-origin:left;opacity:.15;transition:opacity .4s}.roofline .memory-roof{left:17%;bottom:18%;width:52%;transform:rotate(-38deg)}.roofline .compute-roof{left:58%;right:8%;top:23%;background:var(--yellow)}.roofline .point{position:absolute;width:15px;height:15px;border-radius:50%;background:var(--red);opacity:.15}.roofline .p1{left:47%;top:48%}.roofline .p2{left:73%;top:17%;background:var(--green)}.roofline .revealed{opacity:1}.roofline small{position:absolute;bottom:3%;left:35%;color:var(--muted);font-size:8px}.scene footer{display:flex;justify-content:space-between;color:var(--muted);font-size:8px}.controls{grid-column:1/3;grid-row:3;border-top:1px solid #292f32;background:#15181a;display:flex;align-items:center;justify-content:center;gap:7px;position:relative}.controls button{width:35px;height:35px;border:1px solid var(--line);background:var(--panel);color:var(--paper);cursor:pointer}.controls button:hover,.controls button.active{border-color:var(--accent);color:var(--accent)}.build-label,.scene-count{position:absolute;color:var(--muted);font:9px monospace}.build-label{left:20px}.scene-count{right:20px}.inspector{position:absolute;right:0;top:48px;bottom:58px;width:min(390px,88vw);z-index:8;background:#191d1f;border-left:1px solid var(--accent,#36d17d);padding:24px;transform:translateX(100%);transition:transform .3s}.inspector.open{transform:none}.inspector .close{position:absolute;right:13px;top:13px;width:30px;height:30px;border:1px solid var(--line);background:var(--panel);color:var(--paper)}.inspector>b{font-size:8px;color:var(--accent)}.inspector h2{margin:28px 0 18px;font-size:19px}.inspector h3{margin:18px 0 6px;color:var(--muted);font-size:8px;text-transform:uppercase}.inspector code,.inspector p{font:10px/1.55 monospace;color:var(--paper);white-space:pre-wrap}.index-page{min-height:100vh}.index-hero{min-height:62vh;display:grid;grid-template-columns:1.35fr 1fr;gap:70px;align-items:center;padding:8vh 9vw;border-bottom:1px solid var(--line)}.index-hero>div>b{color:var(--green);font-size:10px}.index-hero h1{margin:18px 0;font-size:clamp(42px,6vw,78px);line-height:.96}.index-hero p{max-width:700px;color:var(--muted);font-size:16px;line-height:1.55}.stack-viz{display:grid;gap:9px}.stack-viz i{display:block;padding:17px 20px;border-left:5px solid var(--green);background:var(--panel);font-style:normal;font-size:11px;font-weight:800}.stack-viz i:nth-child(2){border-color:var(--teal)}.stack-viz i:nth-child(3){border-color:var(--violet)}.stack-viz i:nth-child(4){border-color:var(--yellow)}.stack-viz i:nth-child(5){border-color:var(--red)}.index-main{padding:0 9vw 7vh}.stats{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}.stats span{padding:28px 12px;color:var(--muted);font-size:11px}.stats b{display:block;color:var(--paper);font-size:29px;margin-bottom:5px}.part-list{display:grid}.part-row{display:grid;grid-template-columns:80px 1fr 120px;align-items:center;gap:24px;padding:28px 8px;border-bottom:1px solid var(--line);color:var(--paper);text-decoration:none}.part-row:hover{background:#171b1d}.part-row>b{font-size:31px;color:var(--accent)}.part-row strong{display:block;font-size:19px;margin-bottom:7px}.part-row small{display:block;color:var(--muted);font-size:11px}.part-row em{color:var(--muted);font-size:10px;font-style:normal;text-align:right;line-height:1.5}.index-footer{display:flex;justify-content:space-between;padding:24px 9vw;border-top:1px solid var(--line);font-size:10px}.index-footer a{color:var(--green)}.index-footer code{color:var(--muted)}.source-page{min-height:100vh}.source-head{position:sticky;top:0;z-index:3;display:grid;grid-template-columns:180px 1fr minmax(260px,420px);gap:25px;align-items:center;padding:20px 4vw;background:#15181a;border-bottom:1px solid var(--line)}.source-head a{color:var(--green);font-size:10px;font-weight:800;text-decoration:none}.source-head h1{font-size:24px;margin:0}.source-head input{width:100%;padding:11px 13px;background:var(--panel);border:1px solid var(--line);color:var(--paper)}.source-page main{padding:24px 4vw 70px}.source-page table{width:100%;border-collapse:collapse;font-size:10px}.source-page th{text-align:left;color:var(--muted);font-size:8px;text-transform:uppercase;padding:12px 10px;border-bottom:1px solid var(--line)}.source-page td{padding:14px 10px;border-bottom:1px solid #292f32;vertical-align:top}.source-page td:first-child{color:var(--green);font-weight:800}.source-page a{color:var(--paper)}.source-page code{font-size:9px;color:var(--muted)}@keyframes pulse{50%{filter:brightness(1.8)}}@media(max-width:900px){.scene-body{grid-template-columns:1fr}.visual-pane{display:none}.source-pane{min-height:0}.scene{padding:18px}.scene-head h1{font-size:22px}.index-hero{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.source-head{grid-template-columns:1fr}.source-head a{display:none}}@media(max-width:620px){.topbar{grid-template-columns:1fr auto}.part-nav{display:none}.scene-head p{display:none}.scene-body{padding-top:10px}.scene{padding:13px}.code-line code{font-size:8px}.part-row{grid-template-columns:50px 1fr}.part-row em{display:none}.stats{grid-template-columns:1fr 1fr}.index-hero{padding:6vh 7vw}.index-main{padding:0 7vw 7vh}}
`;

const COURSE_JS = String.raw`
(()=>{const scenes=[...document.querySelectorAll('.scene')],dots=[...document.querySelectorAll('.scene-dot')],inspector=document.querySelector('.inspector');if(!scenes.length)return;let index=0,step=0,timer=null;const hash=location.hash.replace('#scene-','');const found=scenes.findIndex(s=>s.id==='scene-'+hash);if(found>=0)index=found;const max=()=>Number(scenes[index].dataset.steps||4);function render(){scenes.forEach((s,i)=>s.classList.toggle('active',i===index));dots.forEach((d,i)=>d.classList.toggle('active',i===index));const scene=scenes[index];document.documentElement.style.setProperty('--accent',getComputedStyle(scene).getPropertyValue('--accent'));scene.querySelectorAll('[data-step]').forEach(el=>{const n=Number(el.dataset.step);el.classList.toggle('revealed',n<=step);el.classList.toggle('current',n===step)});document.querySelector('.build-label').textContent='BUILD '+step+' / '+max();document.querySelector('.scene-count').textContent=String(index+1).padStart(2,'0')+' / '+String(scenes.length).padStart(2,'0');history.replaceState(null,'','#scene-'+scene.id.replace('scene-',''));inspector.classList.remove('open')}function next(){if(step<max())step++;else if(index<scenes.length-1){index++;step=0}else{index=0;step=0}render()}function prev(){if(step>0)step--;else if(index>0){index--;step=max()}render()}function play(){const b=document.querySelector('.play');if(timer){clearInterval(timer);timer=null;b.classList.remove('active');b.innerHTML='&#9654;'}else{timer=setInterval(next,1300);b.classList.add('active');b.innerHTML='&#10074;&#10074;'}}document.querySelector('.next').onclick=next;document.querySelector('.prev').onclick=prev;document.querySelector('.play').onclick=play;document.querySelector('.reset').onclick=()=>{step=0;render()};document.querySelector('.full').onclick=()=>document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();dots.forEach(d=>d.onclick=()=>{index=Number(d.dataset.target);step=0;render()});document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();next()}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();prev()}if(e.key.toLowerCase()==='p')play();if(e.key==='Escape')inspector.classList.remove('open')});document.addEventListener('click',e=>{const el=e.target.closest('.code-line,.d-node');if(!el||!el.classList.contains('revealed'))return;const scene=scenes[index],line=el.dataset.line;inspector.querySelector('h2').textContent=line?'Line '+line:el.dataset.detail;inspector.querySelector('.inspect-source').textContent=scene.querySelector('.source-title a').textContent+(line?':'+line:'');inspector.querySelector('.inspect-detail').textContent=el.dataset.detail||scene.querySelector('.scene-head p').textContent;inspector.querySelector('.inspect-formula').textContent=scene.querySelector('.formula code').textContent;inspector.classList.add('open')});inspector.querySelector('.close').onclick=()=>inspector.classList.remove('open');render()})();
`;

function courseMap(manifest) {
  const sections = course.map((p) => {
    const units = p.units.map((u, i) => `    ${String(i + 1).padStart(2, "0")}. ${u.title}\n        File: \`${u.file}\`\n        Function/anchor: \`${u.anchor.replaceAll("\n", " ")}\`\n        Role: ${u.role}\n        Evidence: \`${u.evidence}\``).join("\n");
    return `- Part ${p.id}: ${p.title} (${p.scenes.length} scenes, ${manifest.decks.find((d) => d.part === p.id).slides} slides)\n${units}`;
  }).join("\n\n");
  return `# Llama / MAX / GPU Masterclass Course Map\n\nRevision: \`${REV}\`\n\nTotal: **${manifest.totalSlides} PowerPoint slides**, **${manifest.totalScenes} animated source scenes**, and **${manifest.sourceFiles} unique source files**.\n\nEach source unit becomes two eight-line scenes. Each scene becomes four progressive PowerPoint builds and one four-step HTML animation. Claims are labeled \`SOURCE\`, \`BOUNDARY\`, \`COMPILE-ONLY\`, or \`GPU-LAB\`.\n\n## Tree\n\n${sections}\n`;
}

function lineByLineAtlas() {
  const sections = course.map((part) => {
    const scenes = part.scenes.map((scene) => {
      const rows = scene.lines.map((entry) => {
        const source = entry.text.trim().replaceAll("`", "\\`");
        return `- \`L${entry.number}\` \`${source}\`\n  ${explainLine(entry.text)}`;
      }).join("\n");
      return `### ${scene.title}\n\n- File: \`${scene.file}\`\n- Source range: \`${scene.lines[0].number}-${scene.lines[7].number}\`\n- Function/anchor: \`${scene.anchor.replaceAll("\n", " ")}\`\n- Evidence: \`${scene.evidence}\`\n- Mathematical model: \`${scene.formula}\`\n\n${rows}`;
    }).join("\n\n");
    return `## Part ${part.id}: ${part.title}\n\n${scenes}`;
  }).join("\n\n");
  return `# Line-by-Line Source Atlas\n\nRevision: \`${REV}\`\n\nThis atlas explains every source line displayed by the 132 animated scenes. The first window captures each callable's contract; the second selects a high-signal implementation block nearby. Follow the file link for the complete source body.\n\n${sections}\n`;
}

function readme(manifest) {
  return `# Llama / MAX / GPU Masterclass\n\nA source-driven course covering MAX graph construction, graph compiler transformations, Mojo GPU kernels, driver offload, target backends, and MAX Serve scheduling.\n\n- PowerPoint: \`decks/\` (${manifest.totalSlides} slides across six files)\n- Browser course: \`html/index.html\`\n- Searchable source index: \`html/source-index.html\`\n- Source/function map: \`COURSE_MAP.md\`\n- Displayed source line explanations: \`LINE_BY_LINE_SOURCE_ATLAS.md\`\n- Reproducible generator: \`src/\`\n\n## Build\n\n\`\`\`bash\ncd murali_docs/artifacts/ModularTechTalk/LLAMA_MAX_GPU_MASTERCLASS/src\nnpm install\nnpm run build\nnpm run check\n\`\`\`\n\nRevision used for source excerpts: \`${REV}\`. GPU timing and achieved overlap remain \`GPU-LAB\` items; source inspection is not presented as measured GPU performance.\n`;
}

async function main() {
  ensure(DECKS); ensure(ASSETS);
  const deckRecords = [];
  for (const part of course) {
    process.stdout.write(`Generating part ${part.id}: ${part.title}\n`);
    const record = await writeDeck(part);
    deckRecords.push({ part: part.id, title: part.title, ...record });
    fs.writeFileSync(path.join(HTML, `part-${String(part.id).padStart(2, "0")}-${part.slug}.html`), partPage(part), "utf8");
  }
  const uniqueFiles = new Set(course.flatMap((p) => p.units.map((u) => u.file)));
  const manifest = {
    revision: REV,
    generatedAt: new Date().toISOString(),
    totalSlides: deckRecords.reduce((n, d) => n + d.slides, 0),
    totalScenes: course.reduce((n, p) => n + p.scenes.length, 0),
    sourceFiles: uniqueFiles.size,
    decks: deckRecords,
    html: ["html/index.html", "html/source-index.html", ...course.map((p) => `html/part-${String(p.id).padStart(2, "0")}-${p.slug}.html`)],
  };
  fs.writeFileSync(path.join(ASSETS, "course.css"), COURSE_CSS, "utf8");
  fs.writeFileSync(path.join(ASSETS, "course.js"), COURSE_JS, "utf8");
  fs.writeFileSync(path.join(HTML, "index.html"), indexPage(manifest), "utf8");
  fs.writeFileSync(path.join(HTML, "source-index.html"), sourceIndexPage(), "utf8");
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  fs.writeFileSync(path.join(OUT, "COURSE_MAP.md"), courseMap(manifest), "utf8");
  fs.writeFileSync(path.join(OUT, "LINE_BY_LINE_SOURCE_ATLAS.md"), lineByLineAtlas(), "utf8");
  fs.writeFileSync(path.join(OUT, "README.md"), readme(manifest), "utf8");
  process.stdout.write(`Generated ${manifest.totalSlides} slides and ${manifest.totalScenes} HTML scenes.\n`);
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
