/* Shared actual-render contrast, bounds and occlusion audit. */
function renderedAudit(){
 const rgb=s=>s.match(/[\d.]+/g)?.map(Number)||[],lum=c=>c.slice(0,3).map(v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;}).reduce((a,v,i)=>a+v*[.2126,.7152,.0722][i],0);
 const ratio=(a,b)=>(Math.max(lum(a),lum(b))+.05)/(Math.min(lum(a),lum(b))+.05);
 const bg=e=>{if(!e)return [255,255,255];const c=rgb(getComputedStyle(e).backgroundColor),back=bg(e.parentElement);return c.slice(0,3).map((v,i)=>v*(c[3]??1)+back[i]*(1-(c[3]??1)));};
 const fails=[],overflow=[],occlusion=[];let samples=0,min=Infinity;
 for(const e of document.body.querySelectorAll("*")){
  if(!e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})||e.closest(":disabled,script,style,option"))continue;
  const text=[...e.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent.trim()).join("");if(!text)continue;
  const b=e.getBoundingClientRect(),s=getComputedStyle(e);if(!b.width)continue;
  let backing=bg(e),ink=rgb(e instanceof SVGElement?s.fill:s.color);
  if(e instanceof SVGTextElement){
   const actor=e.closest(".actor"),owner=e.closest(".owner"),slot=e.closest("[data-slot]");
   if(actor){const shape=actor.querySelector(".actor-paper,.actor-box");backing=rgb(getComputedStyle(shape).fill);
    const a=shape.getBoundingClientRect();if(b.left<a.left-1||b.right>a.right+1||b.top<a.top-1||b.bottom>a.bottom+1)overflow.push({actor:actor.dataset.actor,text,width:b.width,shapeWidth:a.width});
    // Inspect the text's center against later opaque actors in the same scene.
    const x=b.x+b.width/2,y=b.y+b.height/2;
    for(const other of actor.parentElement.querySelectorAll(".actor")){if(other===actor||!(actor.compareDocumentPosition(other)&Node.DOCUMENT_POSITION_FOLLOWING)||other.hidden||other.style.opacity<.95)continue;const box=other.querySelector(".actor-paper,.actor-box").getBoundingClientRect();if(x>box.left+2&&x<box.right-2&&y>box.top+2&&y<box.bottom-2)occlusion.push({actor:actor.dataset.actor,by:other.dataset.actor,text});}
   }else if(owner)backing=rgb(getComputedStyle(owner.querySelector("rect")).fill);
   else if(slot)backing=rgb(getComputedStyle(slot.querySelector("rect")).fill);
   else {backing=[246,249,252];for(const shape of e.closest("svg").querySelectorAll("rect")){if(!(shape.compareDocumentPosition(e)&Node.DOCUMENT_POSITION_FOLLOWING)||!shape.checkVisibility({checkVisibilityCSS:true}))continue;const a=shape.getBoundingClientRect();if(b.x+b.width/2>=a.left&&b.x+b.width/2<=a.right&&b.y+b.height/2>=a.top&&b.y+b.height/2<=a.bottom){const c=rgb(getComputedStyle(shape).fill);if(c.length>=3)backing=c;}}}
  }
  const r=ratio(ink,backing),large=parseFloat(s.fontSize)>=24||(parseFloat(s.fontSize)>=18.66&&Number(s.fontWeight)>=700),threshold=large?3:4.5;samples++;min=Math.min(min,r);if(r+.01<threshold)fails.push({text,ratio:r,threshold});
 }
 for(const e of document.querySelectorAll("button:not(:disabled),select,input:not([type=range]),.wire path")){
  if(!e.checkVisibility({checkVisibilityCSS:true}))continue;const s=getComputedStyle(e),svg=e instanceof SVGElement,ink=rgb(svg?s.stroke:s.borderTopColor),back=svg?[246,249,252]:bg(e.parentElement);const r=ratio(ink,back);if(r+.01<3)fails.push({text:svg?"wire":e.textContent.slice(0,25)+" outline",ratio:r,threshold:3});
 }
 return {samples,minimum:Number(min.toFixed(3)),fails,overflow,occlusion,documentOverflow:document.documentElement.scrollWidth>innerWidth+1};
}
module.exports=renderedAudit;
