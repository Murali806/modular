/* Movable windows shared by the add-one and standalone compiler readers. */
(function(root) {
  'use strict';
  const records=new WeakMap(),windows=new Set();
  let serial=0,z=95;
  const gap=12;
  const visible=card=>card instanceof HTMLDialogElement?card.open:!card.hidden;
  const held=card=>records.get(card)?.held===true;
  const bounds=()=>({width:Math.max(1,innerWidth-2*gap),height:Math.max(1,innerHeight-2*gap)});
  const rect=card=>{const r=card.getBoundingClientRect();return {left:r.left,top:r.top,width:r.width,height:r.height};};
  function fit(card,r) {
    const limit=bounds(),width=Math.min(limit.width,Math.max(Math.min(320,limit.width),r.width)),height=Math.min(limit.height,Math.max(Math.min(140,limit.height),r.height));
    Object.assign(card.style,{width:width+'px',height:height+'px',maxWidth:limit.width+'px',maxHeight:limit.height+'px',left:Math.max(gap,Math.min(innerWidth-width-gap,r.left))+'px',top:Math.max(gap,Math.min(innerHeight-height-gap,r.top))+'px',margin:'0'});
  }
  function buttons(record) {
    record.card.dataset.cardHeld=String(record.held);
    record.card.dataset.cardExpanded=String(record.expanded);
    record.hold.textContent=record.held?'Held':'Hold';
    record.hold.setAttribute('aria-pressed',String(record.held));
    record.hold.title=record.held?'Keep open until Close; click to release':'Keep this card open until Close';
    record.expand.textContent=record.expanded?'Restore':'Expand';
    record.expand.setAttribute('aria-pressed',String(record.expanded));
    record.expand.title=record.expanded?'Restore the previous size':'Expand this card to fit the viewport';
  }
  function register(card,{header,closeButton,close,body}={}) {
    if(records.has(card))return records.get(card);
    header ||= card.querySelector('.call-explanation-card-header,.card-toolbar,.source-toolbar');
    closeButton ||= header?.querySelector('[data-close-call-explanation],[data-close-card],[data-close-legend],#source-close,[data-window-close]');
    if(!header||!closeButton)return null;
    card.classList.add('reader-card-window');
    card.dataset.cardHeld='false';card.dataset.cardExpanded='false';
    header.classList.add('reader-window-header');header.tabIndex=0;
    header.title='Drag to move this card; use arrow keys when this header is focused';
    if(!body){
      body=card.querySelector(':scope > .call-explanation-card-body,:scope > .reader-window-content');
      if(!body){body=document.createElement('div');for(const child of [...card.childNodes])if(child!==header)body.append(child);card.append(body);}
    }
    body.classList.add('reader-window-content');
    let actions=header.querySelector('.reader-window-actions');
    if(!actions){actions=document.createElement('div');actions.className='reader-window-actions';header.append(actions);}
    function button(key,label){let b=actions.querySelector('[data-window-'+key+']');if(!b){b=document.createElement('button');b.type='button';b.dataset['window'+key[0].toUpperCase()+key.slice(1)]='';b.textContent=label;actions.append(b);}return b;}
    const holdButton=button('hold','Hold'),expandButton=button('expand','Expand');
    closeButton.dataset.windowClose='';closeButton.textContent='Close';closeButton.title='Close this card';actions.append(closeButton);
    let handle=card.querySelector(':scope > .reader-window-resize');
    if(!handle){handle=document.createElement('button');handle.type='button';handle.className='reader-window-resize';handle.textContent='◢';handle.setAttribute('aria-label','Resize card');handle.title='Drag to resize; arrow keys change width or height';card.append(handle);}
    const record={card,header,body,closeButton,hold:holdButton,expand:expandButton,held:false,manual:false,expanded:false,restore:null,close};
    records.set(card,record);windows.add(card);buttons(record);
    holdButton.addEventListener('click',()=>{record.held=!record.held;buttons(record);});
    expandButton.addEventListener('click',()=>{
      if(record.expanded){record.expanded=false;fit(card,record.restore||rect(card));}
      else{record.restore=rect(card);record.expanded=true;fit(card,{left:gap,top:gap,width:innerWidth-2*gap,height:innerHeight-2*gap});}
      record.manual=true;buttons(record);
    });
    if(close)closeButton.addEventListener('click',close);
    card.addEventListener('pointerdown',()=>{card.style.zIndex=String(++z);});
    card.addEventListener('focusin',()=>{card.style.zIndex=String(++z);});
    function start(event,resizing){
      if(event.button!==0||(!resizing&&event.target.closest('button,a,input,textarea,select,[contenteditable]')))return;
      event.preventDefault();record.manual=true;
      if(resizing){record.expanded=false;buttons(record);}
      const element=resizing?handle:header,initial=rect(card),x=event.clientX,y=event.clientY;
      fit(card,initial);element.setPointerCapture(event.pointerId);card.dataset.cardDragging='true';
      const move=e=>{if(e.pointerId!==event.pointerId)return;const dx=e.clientX-x,dy=e.clientY-y;fit(card,resizing?{...initial,width:initial.width+dx,height:initial.height+dy}:{...initial,left:initial.left+dx,top:initial.top+dy});};
      const stop=e=>{if(e.pointerId!==event.pointerId)return;card.dataset.cardDragging='false';element.removeEventListener('pointermove',move);element.removeEventListener('pointerup',stop);element.removeEventListener('pointercancel',stop);element.removeEventListener('lostpointercapture',stop);if(element.hasPointerCapture(event.pointerId))element.releasePointerCapture(event.pointerId);};
      element.addEventListener('pointermove',move);element.addEventListener('pointerup',stop);element.addEventListener('pointercancel',stop);element.addEventListener('lostpointercapture',stop);
    }
    header.addEventListener('pointerdown',e=>start(e,false));handle.addEventListener('pointerdown',e=>start(e,true));
    function keyboard(event,resizing){
      if(event.target!==(resizing?handle:header)||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
      event.preventDefault();const r=rect(card),step=event.shiftKey?40:20,dx=event.key==='ArrowRight'?step:event.key==='ArrowLeft'?-step:0,dy=event.key==='ArrowDown'?step:event.key==='ArrowUp'?-step:0;
      record.manual=true;if(resizing){record.expanded=false;buttons(record);}
      fit(card,resizing?{...r,width:r.width+dx,height:r.height+dy}:{...r,left:r.left+dx,top:r.top+dy});
    }
    header.addEventListener('keydown',e=>keyboard(e,false));handle.addEventListener('keydown',e=>keyboard(e,true));
    card.addEventListener('cancel',e=>{if(record.held)e.preventDefault();});
    return record;
  }
  function reset(card){const r=records.get(card);if(!r)return;r.held=false;r.manual=false;r.expanded=false;r.restore=null;for(const property of ['left','top','width','height','maxWidth','maxHeight','margin'])card.style[property]='';buttons(r);}
  function park(card){
    const r=records.get(card);if(!r?.held||!visible(card)||card instanceof HTMLDialogElement)return;
    const geometry=rect(card),copy=card.cloneNode(true),suffix='-held-'+(++serial),ids=new Map();
    for(const element of [copy,...copy.querySelectorAll('[id]')])if(element.id){const id=element.id;ids.set(id,id+suffix);element.id=id+suffix;}
    for(const element of [copy,...copy.querySelectorAll('*')]){
      for(const attr of ['aria-labelledby','aria-describedby','aria-controls','for'])if(element.hasAttribute(attr))element.setAttribute(attr,element.getAttribute(attr).split(' ').map(id=>ids.get(id)||id).join(' '));
      const hash=element.getAttribute('href');if(hash?.startsWith('#')&&ids.has(hash.slice(1)))element.setAttribute('href','#'+ids.get(hash.slice(1)));
      if(element.dataset.callTreeId)element.dataset.callTreeId+=suffix;
    }
    copy.dataset.heldWindow='true';copy.hidden=false;
    (document.fullscreenElement||document.body).append(copy);
    const duplicate=register(copy,{close:()=>{windows.delete(copy);copy.remove();}});
    duplicate.held=true;duplicate.manual=true;duplicate.expanded=r.expanded;duplicate.restore=r.restore;buttons(duplicate);fit(copy,geometry);copy.style.zIndex=String(++z);
    reset(card);
  }
  function opening(card){const r=records.get(card);if(!r)return;if(card instanceof HTMLDialogElement)return;park(card);if(!visible(card))reset(card);}
  function positioned(card){const r=records.get(card);if(!r||!visible(card))return;if(r.expanded)fit(card,{left:gap,top:gap,width:innerWidth-2*gap,height:innerHeight-2*gap});else if(r.manual||r.held)fit(card,rect(card));}
  function followsTrigger(card){const r=records.get(card);return !r||(!r.manual&&!r.expanded&&!r.held);}
  function openSource(card){if(records.has(card)&&!card.open)reset(card);}
  function sourceShown(card){const record=records.get(card);if(record){if(record.manual||record.held){positioned(card);return;}const r=rect(card);fit(card,{...r,left:(innerWidth-r.width)/2,top:(innerHeight-r.height)/2});}}
  document.addEventListener('click',e=>{for(const card of [...windows])if(card.dataset.heldWindow&&!held(card)&&!card.contains(e.target)){windows.delete(card);card.remove();}});
  document.addEventListener('keydown',e=>{if(e.key!=='Escape'||document.getElementById('source-dialog')?.open)return;const copies=[...windows].filter(card=>card.dataset.heldWindow&&!held(card)&&visible(card)).sort((a,b)=>Number(b.style.zIndex)-Number(a.style.zIndex));if(copies.length){e.preventDefault();windows.delete(copies[0]);copies[0].remove();}});
  addEventListener('resize',()=>{for(const card of windows)positioned(card);});
  document.addEventListener('fullscreenchange',()=>{for(const card of windows)if(!(card instanceof HTMLDialogElement)){(document.fullscreenElement||document.body).append(card);if(visible(card))fit(card,rect(card));}});
  root.CompilerCardWindows={register,isHeld:held,opening,closed:reset,positioned,followsTrigger,openSource,sourceShown};
})(window);
