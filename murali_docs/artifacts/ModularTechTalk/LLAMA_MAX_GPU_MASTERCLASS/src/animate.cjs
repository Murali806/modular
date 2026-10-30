"use strict";

const AdmZip = require("adm-zip");

const SLIDE_RE = /^ppt\/slides\/slide(\d+)\.xml$/;
const SHAPE_RE = /<p:cNvPr\s+id="(\d+)"\s+name="([^"]*)"/g;
const ANIM_RE = /^anim-s(\d+)-g(\d+)-/;

function clickGroup(shapeIds, nextId) {
  const outerId = nextId;
  const innerId = nextId + 1;
  const clickId = nextId + 2;
  nextId += 3;
  const sets = shapeIds.map((shapeId) => {
    const behaviorId = nextId;
    nextId += 1;
    return `<p:set><p:cBhvr><p:cTn id="${behaviorId}" dur="1" fill="hold">`
      + '<p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>'
      + `<p:tgtEl><p:spTgt spid="${shapeId}"/></p:tgtEl>`
      + '<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>'
      + '</p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>';
  }).join("");
  return {
    nextId,
    xml: `<p:par><p:cTn id="${outerId}" fill="hold">`
      + '<p:stCondLst><p:cond delay="indefinite"/></p:stCondLst><p:childTnLst>'
      + `<p:par><p:cTn id="${innerId}" fill="hold">`
      + '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
      + `<p:par><p:cTn id="${clickId}" presetID="1" presetClass="entr" presetSubtype="0" fill="hold" nodeType="clickEffect">`
      + '<p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
      + sets
      + '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
      + '</p:childTnLst></p:cTn></p:par>',
  };
}

function timingXml(groups) {
  let nextId = 3;
  const effects = [];
  for (const group of [...groups.keys()].sort((a, b) => a - b)) {
    const result = clickGroup(groups.get(group), nextId);
    nextId = result.nextId;
    effects.push(result.xml);
  }
  return '<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
    + '<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
    + effects.join("")
    + '</p:childTnLst></p:cTn>'
    + '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
    + '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
    + '</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>';
}

function transitionXml(slideNumber) {
  const variants = [
    '<p:transition spd="fast" advClick="1"><p:fade/></p:transition>',
    '<p:transition spd="fast" advClick="1"><p:push dir="l"/></p:transition>',
    '<p:transition spd="fast" advClick="1"><p:wipe dir="r"/></p:transition>',
  ];
  return variants[(slideNumber - 1) % variants.length];
}

function injectAnimations(file) {
  const zip = new AdmZip(file);
  const report = [];
  const slideEntries = zip.getEntries()
    .filter((entry) => SLIDE_RE.test(entry.entryName))
    .sort((a, b) => Number(a.entryName.match(SLIDE_RE)[1]) - Number(b.entryName.match(SLIDE_RE)[1]));

  for (const entry of slideEntries) {
    const slideNumber = Number(entry.entryName.match(SLIDE_RE)[1]);
    let xml = entry.getData().toString("utf8");
    const groups = new Map();
    const names = new Map();
    for (const match of xml.matchAll(SHAPE_RE)) {
      const shapeId = match[1];
      const name = match[2];
      const animation = name.match(ANIM_RE);
      if (!animation) continue;
      const namedSlide = Number(animation[1]);
      const group = Number(animation[2]);
      if (namedSlide !== slideNumber) {
        throw new Error(`Slide ${slideNumber}: animation object ${name} targets slide ${namedSlide}`);
      }
      if (!groups.has(group)) groups.set(group, []);
      if (!names.has(group)) names.set(group, []);
      groups.get(group).push(shapeId);
      names.get(group).push(name);
    }

    xml = xml.replace(/<p:timing>.*?<\/p:timing>/gs, "");
    xml = xml.replace(/<p:transition.*?<\/p:transition>/gs, "");
    const transition = transitionXml(slideNumber);
    if (xml.includes("<p:clrMapOvr")) xml = xml.replace("<p:clrMapOvr", `${transition}<p:clrMapOvr`);
    else xml = xml.replace("</p:sld>", `${transition}</p:sld>`);

    if (groups.size) {
      const ordered = [...groups.keys()].sort((a, b) => a - b);
      const expected = Array.from({ length: ordered.at(-1) }, (_, index) => index + 1);
      if (ordered.join(",") !== expected.join(",")) {
        throw new Error(`Slide ${slideNumber}: non-contiguous animation groups ${ordered.join(",")}`);
      }
      xml = xml.replace("</p:sld>", `${timingXml(groups)}</p:sld>`);
      report.push({
        slide: slideNumber,
        groups: ordered.map((group) => ({
          group,
          shapeIds: groups.get(group),
          shapeNames: names.get(group),
        })),
      });
    }
    zip.updateFile(entry.entryName, Buffer.from(xml, "utf8"));
  }
  zip.writeZip(file);
  return { slides: report, transitionedSlides: slideEntries.length };
}

module.exports = { injectAnimations };
