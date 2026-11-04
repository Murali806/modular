/* Presentation of OS roles and event order; no load or hardware-rate model. */
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("component-data").textContent);
  const $ = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const records = new Map([...document.querySelectorAll("[data-reading]")].map(n => [n.dataset.reading, n]));
  const players = new Map();
  const card = $("reading-card");
  let cardTrigger = null, cardKey = null, held = false, touchTrigger = null, touchAt = -Infinity;
  let raf = null;
  document.documentElement.classList.add("js");
  document.querySelectorAll("[data-interactive]").forEach(n => n.hidden = false);
  document.querySelectorAll("[data-card]").forEach(n => {
    n.setAttribute("aria-controls", "reading-card"); n.setAttribute("aria-expanded", "false");
  });

  function focusQuietly(trigger) {
    if (!trigger?.isConnected || !trigger.checkVisibility()) return;
    trigger.dataset.dismissFocus = "true";
    trigger.focus({preventScroll:true});
    queueMicrotask(() => delete trigger.dataset.dismissFocus);
  }
  function closeCard(restore = false) {
    const trigger = cardTrigger;
    trigger?.setAttribute("aria-expanded", "false");
    card.hidden = true; cardKey = null; cardTrigger = null; held = false;
    card.dataset.held = "false"; $("card-hold").setAttribute("aria-pressed", "false");
    if (restore) focusQuietly(trigger);
  }
  function hold(value) {
    held = value; card.dataset.held = String(value);
    $("card-hold").setAttribute("aria-pressed", String(value));
    $("card-hold").textContent = value ? "Release card" : "Hold card";
  }
  function positionCard(trigger) {
    if (innerWidth <= 560) {card.style.removeProperty("left");card.style.removeProperty("right");return;}
    const rect = trigger.getBoundingClientRect();
    const leftSide = innerWidth > 760 && rect.left + rect.width/2 > innerWidth/2;
    card.style.left = leftSide ? "24px" : "auto";
    card.style.right = leftSide ? "auto" : "24px";
  }
  function showCard(trigger, explicit = false, enter = false) {
    const key = trigger.dataset.card;
    if (!records.has(key) || (!explicit && held)) return;
    if (cardKey !== key || card.hidden) {
      cardTrigger?.setAttribute("aria-expanded", "false");
      const record = records.get(key);
      $("card-title").textContent = record.querySelector(":scope>summary").textContent;
      const body = record.querySelector(":scope>.reading-body").cloneNode(true);
      body.querySelectorAll("[id]").forEach(n => n.removeAttribute("id"));
      $("card-content").replaceChildren(body); $("card-content").scrollTop = 0;
      cardKey = key; card.dataset.key = key; hold(false);
    }
    if (cardTrigger !== trigger) cardTrigger?.setAttribute("aria-expanded", "false");
    cardTrigger = trigger; trigger.setAttribute("aria-expanded", "true"); card.hidden = false;
    positionCard(trigger);
    if (enter) {hold(true); card.focus({preventScroll:true});}
  }
  $("card-close").addEventListener("click", () => closeCard(true));
  window.addEventListener("resize",()=>{if(cardTrigger&&!card.hidden)positionCard(cardTrigger);});
  $("card-hold").addEventListener("click", () => hold(!held));
  document.addEventListener("pointerdown", e => {
    if (e.pointerType === "touch") {touchTrigger = e.target.closest("[data-card]"); touchAt = performance.now();}
  });
  document.addEventListener("pointerover", e => {
    const trigger = e.target.closest("[data-card]");
    if (e.pointerType !== "touch" && trigger && !trigger.contains(e.relatedTarget)) showCard(trigger);
  });
  document.addEventListener("focusin", e => {
    const trigger = e.target.closest("[data-card]");
    if (trigger === touchTrigger && performance.now() - touchAt < 700) return;
    if (trigger && !trigger.dataset.dismissFocus) showCard(trigger);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !card.hidden && !$("source-dialog").open) {e.preventDefault(); closeCard(true); return;}
    const trigger = e.target.closest("[data-card]");
    if (trigger && (e.key === " " || (e.key === "Enter" && trigger instanceof SVGElement))) {
      e.preventDefault(); selectFlowStep(trigger); showCard(trigger, true, true);
    }
  });
  document.addEventListener("click", e => {
    const trigger = e.target.closest("[data-card]");
    if (trigger) {e.preventDefault(); selectFlowStep(trigger); showCard(trigger, true, true);}
  });

  function frameCount(p) {return p.kind === "flow" ? p.data.steps.length : p.data.scenarios[p.scenario].path.length;}
  function setPlaying(p, value) {
    p.playing = value; p.nextAt = performance.now() + 1700 / p.speed;
    const button = p.node.querySelector('[data-action="play"]');
    button.textContent = value ? "Pause" : "Play"; button.setAttribute("aria-pressed", String(value));
    if (value) ensureAnimation();
  }
  function pauseAll() {players.forEach(p => setPlaying(p, false));}
  function setFrame(p, index, animate = false) {
    p.index = Math.max(0, Math.min(frameCount(p)-1, index));
    let path = null;
    if (p.kind === "flow") {
      const step = p.data.steps[p.index];
      p.node.querySelectorAll(".sequence-event").forEach((n, i) => {
        n.classList.toggle("active", i === p.index); n.setAttribute("aria-hidden", String(i !== p.index));
      });
      p.node.querySelectorAll("[data-actor]").forEach(n => n.dataset.current = String([step.sender,step.receiver].includes(n.dataset.actor)));
      p.node.querySelectorAll("[data-story-step]").forEach(n => n.setAttribute("aria-pressed",String(Number(n.dataset.storyStep)===p.index)));
      p.node.querySelector("[data-current-title]").textContent = step.title;
      p.node.querySelector("[data-current-owner]").textContent = step.owner;
      p.node.querySelector("[data-current-detail]").textContent = step.interaction;
      p.node.querySelector("[data-current-result]").textContent = step.result;
      p.node.querySelector("[data-owner-states]").innerHTML = Object.entries(step.states).map(([k,v]) => '<span><b>'+escape(k)+'</b> '+escape(v)+'</span>').join("");
      path = p.node.querySelector(".sequence-event.active [data-motion-path]");
      p.node.dataset.frame = String(p.index);
      if (step.states.HTTP) p.node.dataset.http = step.states.HTTP;
    } else {
      const route = p.data.scenarios[p.scenario].path;
      const current = route[p.index], state = p.data.states[current];
      const previous = p.index ? route[p.index-1] : null;
      const edgeKey = previous ? previous + "-" + current : null;
      const edge = previous ? p.data.edges.find(e => e.start === previous && e.end === current) : null;
      p.node.querySelectorAll("[data-state]").forEach(n => {
        n.classList.toggle("active", n.dataset.state === current);
        n.classList.toggle("visited", route.slice(0,p.index).includes(n.dataset.state) && n.dataset.state !== current);
      });
      p.node.querySelectorAll("[data-edge]").forEach(n => n.classList.toggle("active",n.dataset.edge===edgeKey));
      p.node.querySelector("[data-machine-state]").textContent = state.label;
      p.node.querySelector("[data-machine-detail]").textContent = edge ? edge.detail : state.meaning;
      const label = p.node.querySelector("[data-current-transition]");
      label.textContent = edge ? edge.label : "Inspect " + state.label;
      label.dataset.card = edge ? "edge:" + p.data.id + ":" + edgeKey : "state:" + p.data.id + ":" + current;
      label.setAttribute("aria-expanded", "false");
      p.node.dataset.state = current;
      path = edgeKey ? p.node.querySelector('[data-edge="'+edgeKey+'"]') : null;
    }
    p.node.querySelector("[data-progress]").textContent = (p.index+1) + " / " + frameCount(p);
    p.node.querySelector('[data-action="back"]').disabled = p.index === 0;
    p.node.querySelector('[data-action="next"]').disabled = p.index === frameCount(p)-1;
    p.packet = animate && path && !reduced.matches ? {path, started:performance.now(), duration:900/p.speed} : null;
    p.node.querySelector(".sequence-packet").classList.toggle("visible",!!p.packet);
    if (p.packet) ensureAnimation();
    if (p.index === frameCount(p)-1 && p.playing) setPlaying(p, false);
  }
  function selectFlowStep(trigger) {
    if (trigger.dataset.storyStep === undefined) return;
    const node = trigger.closest("[data-flow]");
    const p = players.get("flow:" + node.dataset.flow);
    setPlaying(p, false); setFrame(p, Number(trigger.dataset.storyStep), true);
  }
  function ensureAnimation() {if (raf === null) raf=requestAnimationFrame(tick);}
  function tick(now) {
    raf = null; let live = false;
    for (const p of players.values()) {
      if (p.playing && now >= p.nextAt) {
        if (p.index < frameCount(p)-1) setFrame(p, p.index+1, true); else setPlaying(p, false);
        p.nextAt = now + 1700 / p.speed;
      }
      if (p.packet) {
        const t = Math.min(1,(now-p.packet.started)/p.packet.duration);
        const at = p.packet.path.getPointAtLength(p.packet.path.getTotalLength()*t);
        p.node.querySelector(".sequence-packet").setAttribute("transform",`translate(${at.x} ${at.y})`);
        if (t >= 1) {p.packet=null; p.node.querySelector(".sequence-packet").classList.remove("visible");}
      }
      if (p.playing || p.packet) live = true;
    }
    if (live) ensureAnimation();
  }
  for (const [kind, list] of [["flow",data.flows],["machine",data.machines]]) {
    for (const record of list) {
      const node = document.querySelector('[data-'+kind+'="'+record.id+'"]');
      const p = {kind, data:record, node, index:0, speed:1, playing:false, packet:null, scenario:record.scenarios ? Object.keys(record.scenarios)[0] : null};
      players.set(kind+":"+record.id,p);
      node.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", () => {
        const action = button.dataset.action;
        if (action === "play") {if (p.index===frameCount(p)-1) setFrame(p,0); setPlaying(p,!p.playing); if(p.playing)setFrame(p,p.index,true);}
        else {setPlaying(p,false); setFrame(p,action==="back"?p.index-1:action==="replay"?0:p.index+1,true);}
      }));
      node.querySelector("[data-speed]").addEventListener("change", e => {p.speed=Number(e.target.value);p.nextAt=performance.now()+1700/p.speed;});
      if (kind === "machine") node.querySelector("[data-scenario]").addEventListener("change", e => {setPlaying(p,false);p.scenario=e.target.value;setFrame(p,0);});
      if (node instanceof HTMLDetailsElement) node.addEventListener("toggle",()=>{if(!node.open)setPlaying(p,false);});
      setFrame(p,0);
    }
  }
  reduced.addEventListener("change",()=>{if(reduced.matches)players.forEach(p=>{p.packet=null;p.node.querySelector(".sequence-packet").classList.remove("visible");});});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pauseAll();});

  const sourceDialog = $("source-dialog");
  let sourceRef = null, sourceReturn = null, sourceMatches = [], matchIndex = -1;
  function showSource(key, trigger) {
    sourceRef = data.references[key]; if (!sourceRef) return;
    sourceReturn = card.contains(trigger) ? cardTrigger : trigger;
    closeCard(false); pauseAll();
    $("source-title").textContent = sourceRef.symbol;
    $("source-location").textContent = sourceRef.file + " · L" + sourceRef.start + "–L" + sourceRef.end + " · selected L" + sourceRef.line + " · " + data.revision.slice(0,12);
    $("source-download").href = "../sources/" + sourceRef.file;
    $("source-search").value = "";
    const lines = data.files[sourceRef.file].text.split("\n"); if(lines.at(-1)==="")lines.pop();
    $("source-file").innerHTML = lines.map((text,i)=>'<span class="source-line'+(i+1>=sourceRef.start&&i+1<=sourceRef.end?' definition':'')+(i+1===sourceRef.line?' selected':'')+'" data-line="'+(i+1)+'"><span class="line-number" aria-hidden="true">'+(i+1)+'</span><span class="source-text">'+escape(text)+'</span></span>').join("\n");
    findSource(); sourceDialog.showModal();
    requestAnimationFrame(()=>$("source-file").querySelector(".selected").scrollIntoView({block:"center"}));
  }
  function findSource() {
    const query = $("source-search").value.toLowerCase(); sourceMatches=[]; matchIndex=-1;
    $("source-file").querySelectorAll(".source-line").forEach(n=>{
      n.classList.remove("search-current");
      if(query&&n.querySelector(".source-text").textContent.toLowerCase().includes(query))sourceMatches.push(n);
    });
    $("source-count").textContent = query ? sourceMatches.length + " matching lines" : "";
    $("source-next").disabled = sourceMatches.length===0;
    if(sourceMatches.length)nextMatch();
  }
  function nextMatch() {
    if(!sourceMatches.length)return;
    sourceMatches.forEach(n=>n.classList.remove("search-current"));
    matchIndex=(matchIndex+1)%sourceMatches.length;
    sourceMatches[matchIndex].classList.add("search-current");sourceMatches[matchIndex].scrollIntoView({block:"center"});
    $("source-count").textContent=(matchIndex+1)+" / "+sourceMatches.length+" matching lines";
  }
  async function copyDefinition() {
    if(!sourceRef)return;
    try{await navigator.clipboard.writeText(sourceRef.code);}
    catch(_){const area=document.createElement("textarea");area.value=sourceRef.code;area.style.cssText="position:fixed;left:-10000px;top:0";sourceDialog.append(area);area.select();const copied=document.execCommand("copy");area.remove();if(!copied){$("copy-status").textContent="Select the source to copy it.";$("copy-status").hidden=false;return;}}
    $("copy-status").textContent="Complete definition copied.";$("copy-status").hidden=false;setTimeout(()=>$("copy-status").hidden=true,2000);
  }
  document.addEventListener("click",e=>{const link=e.target.closest("a[data-ref]");if(link){e.preventDefault();showSource(link.dataset.ref,link);}});
  $("source-close").addEventListener("click",()=>sourceDialog.close());
  sourceDialog.addEventListener("close",()=>focusQuietly(sourceReturn));
  $("source-search").addEventListener("input",findSource);$("source-next").addEventListener("click",nextMatch);$("source-copy").addEventListener("click",copyDefinition);

  let printStates = null;
  window.addEventListener("beforeprint",()=>{pauseAll();closeCard(false);printStates=[...document.querySelectorAll("details")].map(n=>[n,n.open]);printStates.forEach(([n])=>n.open=true);});
  window.addEventListener("afterprint",()=>{printStates?.forEach(([n,open])=>n.open=open);printStates=null;});
  $("print-page").addEventListener("click",()=>window.print());
  function followHash() {
    const id=location.hash.slice(1); if(!id)return;
    const target=document.getElementById(id);if(!target)return;
    for(let n=target;n;n=n.parentElement)if(n instanceof HTMLDetailsElement)n.open=true;
  }
  window.addEventListener("hashchange",followHash);followHash();
})();
