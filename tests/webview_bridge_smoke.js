/* Run with `node tests/webview_bridge_smoke.js`.
 * Simulate pywebview registering its API in two phases (object, then functions).
 * No GUI, Python runtime, HTTP listener or third-party Node modules required.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../web/desktop_webview.js'), 'utf8');

async function scenario(eventDispatched) {
  const elements = new Map();
  class Element {
    constructor(id) {
      this.id = id;
      this.value = '';
      this.textContent = '';
      this.style = {};
      this.files = [];
    }
    addEventListener() {}
    replaceChildren() {}
    append() {}
  }
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new Element(id));
      return elements.get(id);
    },
    createElement(tag) { return new Element(tag); },
  };
  const events = new Map();
  const window = {
    addEventListener(name, cb) {
      const listeners = events.get(name) || new Set();
      listeners.add(cb);
      events.set(name, listeners);
    },
    removeEventListener(name, cb) { events.get(name)?.delete(cb); },
    dispatchEvent(name) { for (const cb of events.get(name) || []) cb(); },
  };
  const context = vm.createContext({
    window, document,
    setTimeout, clearTimeout, setInterval: () => 1,
    Promise, Date, Number, String, Error,
  });
  vm.runInContext(source, context, {filename: 'desktop_webview.js'});

  // Stage 1: framework exposed, but API functions have NOT been registered yet.
  window.pywebview = {api: {}};
  if (eventDispatched) window.dispatchEvent('pywebviewready');
  await new Promise(resolve => setTimeout(resolve, 60));
  assert.notEqual(document.getElementById('feedback').textContent, '準備完了 · Fedpara Desktop（WebView / Python）');

  // Stage 2: methods appear asynchronously, possibly without another event.
  const methods = ['defaults','new_room','save_room','join_room','register_code',
    'start_training','training_status','stop_training'];
  for (const method of methods) {
    window.pywebview.api[method] = async () => method === 'defaults'
      ? {room:'room-test',model:'unit/test',revision:'main',task:'causal-lm',
         ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',rank:32,sources:[]}
      : method === 'training_status' ? {running:false,exit_code:null,logs:''} : {};
  }
  if (eventDispatched) window.dispatchEvent('pywebviewready');
  await new Promise(resolve => setTimeout(resolve, 350));
  assert.equal(document.getElementById('room').value, 'room-test');
  assert.match(document.getElementById('feedback').textContent, /準備完了/);
}

(async () => {
  await scenario(true);
  await scenario(false);
  console.log('PASS: event-based and polling-based late Python bridge initialization');
})().catch(error => {console.error(error); process.exitCode = 1;});
