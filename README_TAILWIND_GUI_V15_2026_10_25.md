> **Superseded by [`README_TAILWIND_CDN_V24.md`](README_TAILWIND_CDN_V24.md).** The build-only workflow below documents the historical v15 release and no longer applies to v24.

# Fedpara v15 — Tailwind CSS desktop GUI

The **native pywebview console** (`python -m patchlib.cli gui`) now uses Tailwind CSS v4.1.10.
This is a **styling-only** migration from v14: FedSum, Gossip/WebRTC, update formats,
random/TinyStories init, ReLoRA, signing and journal behavior are unchanged.

## Quick edit loop

```bash
cd fedpara
npm install
# Edit web/desktop_webview.html and/or web/tailwind.input.css
npm run build:css
python -m patchlib.cli gui
```

There is **no Node, npm, Tailwind, CDN or internet dependency at GUI runtime**:
`web/desktop_webview.css` is a committed, generated artifact. Python's
`patchlib/webview_gui.py::compose_html()` embeds that CSS and the existing
`web/desktop_webview.js` in the local HTML string, preserving the offline CSP.

### Where to change things

- **Layout, spacing, typography of one element**: edit Tailwind utility classes
  directly in `web/desktop_webview.html` (e.g. `px-5`, `lg:grid-cols-2`,
  `text-accent-light`). Run `npm run build:css` after adding class names.
- **Common buttons, input fields, room cards, monitoring graphs**: edit the
  corresponding rule in `web/tailwind.input.css`. These reusable classes use
  Tailwind `@apply` and are also used by dynamically created JavaScript DOM
  nodes (`.room-card`, `.dataset`, `.benchmark-stat`, etc.).
- **Colors and fonts**: change the `@theme` tokens at the top of
  `web/tailwind.input.css` (`--color-paper`, `--color-accent`,
  `--font-console`, `--font-display`, etc.).
- **Behavior, DOM event handlers, Python bridge**: edit
  `web/desktop_webview.js` and `patchlib/webview_gui.py` as before.

**Never edit `web/desktop_webview.css` manually.** It is overwritten by
`npm run build:css`. The bundled `scripts/build_tailwind.mjs` scans static
`class="..."` in HTML and static `className = '...'` strings in JS. For
computed/dynamic classes, define the semantic selector in `@layer components`
with `@apply` (rather than concatenating Tailwind class names at runtime).

## Verification

```bash
npm run check:css
node --check web/desktop_webview.js
node tests/webview_bridge_smoke.js
node tests/webview_click_smoke.js
node tests/webview_room_directory.js
python -m pytest -q tests/test_webview_gui.py tests/test_tailwind_gui.py
```

The same HTML IDs, JavaScript event bindings, and pywebview bridge routes are
preserved. This conversion applies to the **native desktop console** only;
`web/index.html`, `web/dashboard.html`, and `web/patches.html` are separate,
independent frontends and have not been visually migrated in v15.

This design ships with black/dark-green surfaces, emerald-500 accents and
Chicago/Courier fallbacks. No font is downloaded at runtime.
