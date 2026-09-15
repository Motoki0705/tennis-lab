// Lucide icon paths, ISC license: ../../shared/LICENSE-lucide.txt.
const paths = {
  scan: '<path d="M4 7V4h3m10 0h3v3m0 10v3h-3M7 20H4v-3"/><circle cx="12" cy="12" r="3"/>',
  refresh:
    '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  folder:
    '<path d="M20 20H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2Z"/>',
  left: '<path d="m15 18-6-6 6-6"/>',
  right: '<path d="m9 18 6-6-6-6"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  pause: '<path d="M9 4v16M15 4v16"/>',
  first: '<path d="m19 20-9-8 9-8ZM5 19V5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  maximize:
    '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3m0 8v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3"/>',
  download:
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>',
};
export function icon(name) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.scan}</svg>`;
}
export function fillIcons(root = document) {
  for (const el of root.querySelectorAll("[data-icon]"))
    el.innerHTML = icon(el.dataset.icon);
}
