"use strict";

// Run inside a browser page. This checks rendered text colors, not overall
// accessibility conformance. SVG actor text uses its drawn rectangle as backing.
module.exports = function auditRenderedContrast() {
  const styles = new WeakMap();
  const backgrounds = new WeakMap();
  const style = e => {
    if (!styles.has(e)) styles.set(e, getComputedStyle(e));
    return styles.get(e);
  };
  const rgba = value => {
    const n = value.match(/[\d.]+/g)?.map(Number);
    if (!n || n.length < 3) throw new Error(`Unsupported color: ${value}`);
    return [n[0], n[1], n[2], n[3] ?? 1];
  };
  const over = (front, back) => front.slice(0, 3).map((v, i) => v * front[3] + back[i] * (1 - front[3]));
  const background = e => {
    if (!e) return [255, 255, 255];
    if (!backgrounds.has(e)) backgrounds.set(e, over(rgba(style(e).backgroundColor), background(e.parentElement)));
    return backgrounds.get(e);
  };
  const luminance = c => c.slice(0, 3).map(v => {
    v /= 255;
    return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
  }).reduce((v, n, i) => v + n * [.2126, .7152, .0722][i], 0);
  const ratio = (a, b) => {
    const x = luminance(a), y = luminance(b);
    return (Math.max(x, y) + .05) / (Math.min(x, y) + .05);
  };
  const failures = [];
  const checked = [];
  function check(e, color, backing, text, threshold, kind = "text") {
    const contrast = ratio(over(rgba(color), backing), backing);
    const item = {
      element: e.tagName.toLowerCase() + (e.id ? `#${e.id}` : `.${e.getAttribute("class") || ""}`),
      text: text.slice(0, 90), ratio: Number(contrast.toFixed(3)), threshold, kind,
    };
    checked.push(item);
    if (contrast + .001 < threshold) failures.push(item);
  }
  for (const e of document.body.querySelectorAll("*")) {
    if (!e.checkVisibility({checkOpacity:true, checkVisibilityCSS:true}) || e.closest(":disabled")) continue;
    // SVG text can report visibility even inside a display:none group.
    if (e instanceof SVGTextElement && e.getBoundingClientRect().width === 0) continue;
    const s = style(e);
    const text = [...e.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent.trim()).join("");
    const large = parseFloat(s.fontSize) >= 24 || (parseFloat(s.fontSize) >= 18.66 && Number(s.fontWeight) >= 700);
    let backing = background(e);
    if (e.classList.contains("actor-label")) backing = rgba(style(e.closest("svg").querySelector(".actor-box")).fill);
    if (e.matches(".actor-subtitle, .object-hint")) backing = rgba(style(e.closest(".actor").querySelector(".actor-box")).fill);
    if (e.classList.contains("step-number-text")) backing = rgba(style(e.parentElement.querySelector(".step-number-box")).fill);
    if (e.matches(".sequence-packet text")) backing = rgba(style(e.parentElement.querySelector("circle")).fill);
    if (e.matches(".state-label, .state-owner")) backing = rgba(style(e.closest(".state-node").querySelector(".state-box")).fill);
    if (text && !["SCRIPT", "STYLE", "OPTION"].includes(e.tagName)) {
      check(e, e instanceof SVGElement ? s.fill : s.color, backing, text, large ? 3 : 4.5);
    }
    if (["INPUT", "SELECT"].includes(e.tagName)) {
      check(e, s.color, backing, e.value || "input text", 4.5);
      if (e.placeholder && !e.value) check(e, getComputedStyle(e, "::placeholder").color, backing, e.placeholder, 4.5, "placeholder");
    }
    if (e.matches(".call-arrow, .lifeline, .state-edge")) check(e, s.stroke, backing, "diagram line", 3, "graphic");
    if (e.matches(".call-arrowhead")) check(e, s.fill, backing, "diagram arrowhead", 3, "graphic");
    if (e.matches("button, input, select") && parseFloat(s.borderTopWidth) > 0) {
      check(e, s.borderTopColor, background(e.parentElement), "control outline", 3, "control");
    }
    if (e.matches(":focus-visible") && parseFloat(s.outlineWidth) > 0) {
      check(e, s.outlineColor, background(e.parentElement), "keyboard focus", 3, "focus");
    }
  }
  const text = checked.filter(c => c.kind === "text" || c.kind === "placeholder");
  return {samples:checked.length, minimum_text_ratio:Math.min(...text.map(c => c.ratio)), failures};
};
