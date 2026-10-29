# Fedpara GUI: Tailwind v4 CDN (v24)

**All GUI pages (including the diagnostic view) use the browser-based Tailwind engine:**

- `web/desktop_webview.html`: desktop pywebview/Python bridge GUI
- `web/index.html`: WebRTC test page
- `web/dashboard.html`: local HTTP admin console
- `web/patches.html`: P2P patch transfer page
- `patchlib/webview_diagnostic.py`: minimal WebView diagnostic

The Tailwind v4 browser build is loaded from
`https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.1.10`.

**No npm, Node.js, local CSS compiler, generated desktop stylesheet, or bundler is
required** to edit or run the GUI. Edit desktop layout/classes in
`web/desktop_webview.html`, shared component styles and dark emerald theme in
`web/tailwind.cdn.css`, and desktop behavior in `web/desktop_webview.js`.
Dynamic JS-rendered cards use Tailwind `@apply`-based semantic classes defined
in `tailwind.cdn.css`, so DOM changes remain styled.

`python -m patchlib.cli gui` starts the native desktop window as before.
The Python bridge uses a restricted CSP with an explicit CDN exception.

**Important:** Tailwind's Play CDN is intended for development, not production.
The GUI requires network access to load the CDN script each time the window
opens, and will be unstyled if the CDN is blocked/offline. The runtime receives
and executes third-party JS from jsDelivr; deploy only when this trust trade-off
is acceptable. The Python/native model code remains local, and neither model
weights nor datasets are sent to jsDelivr by the Fedpara application, but the
CDN can see the client's HTTP request metadata. For offline production use,
return to a pinned local build with an appropriate CSP.

**Scope:** This is a GUI styling/runtime change only. The P2P protocol,
training model, signed messages, and binary journal remain unchanged.
