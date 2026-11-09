// Trusted, hash-pinned presentation bridge. No token, storage or network API.
(() => {
  'use strict';
  const script = document.currentScript;
  const documentId = script.dataset.readerDocument;
  const revision = script.dataset.readerRevision;
  const hostOrigin = new URL(document.URL).origin;
  const channel = 'explain-reader-v1';
  const blocks = new Map(
    [...document.querySelectorAll('[data-content-id]')].map((el) => [el.dataset.contentId, el]),
  );
  let view = null,
    scheduled = false,
    initialized = false;
  const send = (type, data = {}) =>
    parent.postMessage(
      { channel, type, document_id: documentId, revision, view, ...data },
      hostOrigin,
    );
  const owner = (node) =>
    (node?.nodeType === Node.ELEMENT_NODE ? node : node?.parentElement)?.closest(
      '[data-content-id]',
    );
  const owned = (block, selector) =>
    [...block.querySelectorAll(selector)].filter((el) => owner(el) === block);
  const detailsFor = (block) => owned(block, 'details');
  const inputsFor = (block) => owned(block, 'input,select,textarea');
  const save = (restored = false, missingAnchor = false) => {
    if (!view) return;
    let anchor = null;
    for (const [id, el] of blocks) {
      const rect = el.getBoundingClientRect();
      if (rect.top <= 8 && rect.bottom > 8) {
        anchor = { content_id: id, offset: rect.top };
        break;
      }
      if (!anchor && rect.top >= 0) anchor = { content_id: id, offset: rect.top };
    }
    const states = [];
    for (const [content_id, el] of blocks) {
      states.push({
        content_id,
        details: detailsFor(el).map((item) => item.open),
        inputs: inputsFor(el).map((item) => ({
          value: item.value,
          checked: Boolean(item.checked),
        })),
      });
    }
    send('state', {
      state: { anchor, x: scrollX, y: scrollY, blocks: states },
      restored,
      missing_anchor: missingAnchor,
    });
  };
  const schedule = () => {
    if (scheduled || !initialized) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      save();
    });
  };
  const selectContent = (el) => {
    const block = owner(el);
    if (block && view)
      send('selection', { target: { kind: 'content', content_id: block.dataset.contentId } });
  };
  const selectText = () => {
    const selection = getSelection();
    if (!view || !selection || selection.isCollapsed || selection.rangeCount !== 1) return false;
    const range = selection.getRangeAt(0);
    const block = owner(range.commonAncestorContainer);
    if (!block || !block.contains(range.startContainer) || !block.contains(range.endContainer)) {
      send('selection', { error: 'multiple_blocks' });
      return true;
    }
    const before = range.cloneRange();
    before.selectNodeContents(block);
    before.setEnd(range.startContainer, range.startOffset);
    const start = before.toString().length,
      exact = range.toString();
    if (!exact.trim()) return false;
    const end = start + exact.length,
      text = block.textContent;
    if (text.slice(start, end) !== exact) {
      send('selection', { error: 'dynamic_selection' });
      return true;
    }
    send('selection', {
      target: {
        kind: 'selection',
        content_id: block.dataset.contentId,
        start,
        end,
        exact,
        prefix: Array.from(text.slice(0, start)).slice(-32).join(''),
        suffix: Array.from(text.slice(end)).slice(0, 32).join(''),
      },
    });
    return true;
  };
  for (const el of document.querySelectorAll('h1,h2,h3,h4,h5,h6,[data-figure-id]')) {
    if (!owner(el)) continue;
    el.tabIndex = 0;
    el.addEventListener('click', (event) => {
      if (event.target.closest('input,button,select,textarea,a,summary')) return;
      if (!selectText()) selectContent(el);
    });
    el.addEventListener('keydown', (event) => {
      if (event.target === el && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        selectContent(el);
      }
    });
  }
  document.addEventListener('mouseup', selectText);
  document.addEventListener('keyup', (event) => {
    if (event.key.startsWith('Arrow') || event.key === 'Shift') selectText();
  });
  for (const name of ['input', 'change', 'toggle']) document.addEventListener(name, schedule, true);
  addEventListener('scroll', schedule, { passive: true });
  addEventListener('message', (event) => {
    const message = event.data;
    if (
      event.source !== parent ||
      event.origin !== hostOrigin ||
      !message ||
      message.channel !== channel ||
      message.document_id !== documentId ||
      message.revision !== revision
    )
      return;
    if (
      message.type === 'connect' &&
      !view &&
      typeof message.view === 'string' &&
      /^[a-f0-9]{32}$/.test(message.view)
    ) {
      view = message.view;
      const state = message.state;
      const keep = new Set(Array.isArray(message.unchanged) ? message.unchanged : []);
      if (state && Array.isArray(state.blocks))
        for (const saved of state.blocks) {
          const block = blocks.get(saved.content_id);
          if (!block || !keep.has(saved.content_id)) continue;
          const details = detailsFor(block),
            inputs = inputsFor(block);
          if (Array.isArray(saved.details))
            saved.details.forEach((open, i) => {
              if (details[i] && typeof open === 'boolean') details[i].open = open;
            });
          if (Array.isArray(saved.inputs))
            saved.inputs.forEach((input, i) => {
              if (!inputs[i] || typeof input.value !== 'string') return;
              inputs[i].value = input.value;
              if (typeof input.checked === 'boolean') inputs[i].checked = input.checked;
              inputs[i].dispatchEvent(new Event('input', { bubbles: true }));
              inputs[i].dispatchEvent(new Event('change', { bubbles: true }));
            });
        }
      if (Array.isArray(message.changes))
        for (const id of message.changes) {
          const block = blocks.get(id);
          if (block) {
            block.dataset.readerChanged = 'true';
            block.style.boxShadow = 'inset 3px 0 #617f94';
          }
        }
      const anchor = state?.anchor && blocks.get(state.anchor.content_id);
      // Staging frames are not visible. Edge may suspend their animation frames,
      // so readiness must not depend on requestAnimationFrame. Resources have
      // loaded before 'ready'; this layout read also flushes restored controls.
      if (anchor)
        scrollTo({
          left: state.x,
          top: anchor.getBoundingClientRect().top + scrollY - state.anchor.offset,
          behavior: 'instant',
        });
      else if (state) scrollTo({ left: state.x, top: state.y, behavior: 'instant' });
      initialized = true;
      save(true, Boolean(state?.anchor && !anchor));
    } else if (
      message.type === 'navigate' &&
      message.view === view &&
      blocks.has(message.content_id)
    ) {
      const el = blocks.get(message.content_id);
      for (let ancestor = el.parentElement; ancestor; ancestor = ancestor.parentElement)
        if (ancestor.tagName === 'DETAILS') ancestor.open = true;
      el.tabIndex = -1;
      el.focus({ preventScroll: true });
      el.scrollIntoView({ block: 'start', behavior: 'instant' });
      schedule();
    }
  });
  if (document.readyState === 'complete') send('ready');
  else addEventListener('load', () => send('ready'), { once: true });
})();
