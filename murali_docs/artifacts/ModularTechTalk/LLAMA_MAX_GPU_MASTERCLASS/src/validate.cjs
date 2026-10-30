#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const AdmZip = require("adm-zip");

const OUT = path.resolve(__dirname, "..");
const manifestFile = path.join(OUT, "manifest.json");
if (!fs.existsSync(manifestFile)) throw new Error("Run npm run build before validation.");
const manifest = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
const failures = [];
const assert = (condition, message) => { if (!condition) failures.push(message); };

assert(manifest.totalSlides >= 500, `Expected at least 500 slides, found ${manifest.totalSlides}`);
assert(manifest.decks.length === 6, `Expected six decks, found ${manifest.decks.length}`);
assert(manifest.totalScenes === 132, `Expected 132 scenes, found ${manifest.totalScenes}`);

for (const deck of manifest.decks) {
  const file = path.join(OUT, deck.file);
  assert(fs.existsSync(file), `Missing deck ${deck.file}`);
  if (!fs.existsSync(file)) continue;
  const zip = new AdmZip(file);
  const entries = zip.getEntries();
  const slides = entries.filter((e) => /^ppt\/slides\/slide\d+\.xml$/.test(e.entryName));
  const notes = entries.filter((e) => /^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(e.entryName));
  const transitioned = slides.filter((e) => e.getData().toString("utf8").includes("<p:transition"));
  assert(slides.length === deck.slides, `${deck.file}: manifest ${deck.slides}, archive ${slides.length}`);
  assert(notes.length === slides.length, `${deck.file}: ${notes.length}/${slides.length} slides have notes`);
  assert(transitioned.length === slides.length, `${deck.file}: ${transitioned.length}/${slides.length} slides have transitions`);
}

for (const relative of manifest.html) {
  const file = path.join(OUT, relative);
  assert(fs.existsSync(file), `Missing HTML ${relative}`);
  if (!fs.existsSync(file)) continue;
  const html = fs.readFileSync(file, "utf8");
  assert(html.includes("<!doctype html>"), `${relative}: missing doctype`);
  assert(!/(src|href)=["']https?:\/\//.test(html), `${relative}: external runtime dependency found`);
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const target = match[1];
    if (target.startsWith("#") || target.startsWith("../../../../../")) continue;
    const clean = target.split("#")[0];
    if (!clean) continue;
    assert(fs.existsSync(path.resolve(path.dirname(file), clean)), `${relative}: broken local link ${target}`);
  }
}

const css = fs.readFileSync(path.join(OUT, "html/assets/course.css"), "utf8");
assert(css.includes("@keyframes pulse"), "Course CSS lacks animation keyframes");
assert(!/letter-spacing:\s*-/.test(css), "Negative letter spacing is not allowed");

if (failures.length) {
  process.stderr.write(failures.map((f) => `FAIL: ${f}`).join("\n") + "\n");
  process.exit(1);
}

process.stdout.write(`Validated ${manifest.totalSlides} slides across ${manifest.decks.length} decks and ${manifest.html.length} HTML pages.\n`);
