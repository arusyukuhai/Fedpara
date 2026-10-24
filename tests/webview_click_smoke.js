'use strict';
// Verify the UI responds to clicks before any Python bridge exists.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const code = fs.readFileSync(path.join(__dirname, '../web/desktop_webview.js'), 'utf8');
class E {
  constructor(){this.textContent='';this.value='';this.files=[];this.style={};this.handlers={};}
  addEventListener(event, callback){this.handlers[event]=callback;}
  click(){if(this.handlers.click)this.handlers.click();}
  replaceChildren(){}
  append(){}
}
const byId=new Map();
const document={getElementById(id){if(!byId.has(id))byId.set(id,new E());return byId.get(id);},createElement(){return new E();}};
const listeners={};
const window={addEventListener(type,cb){(listeners[type] ||= []).push(cb);},removeEventListener(){}};
const ctx=vm.createContext({document,window,setTimeout,clearTimeout,setInterval:()=>1,Date,Promise,Number,Error});
vm.runInContext(code,ctx);
assert.match(document.getElementById('uiProbeStatus').textContent,/JavaScript読み込み完了/);
document.getElementById('uiProbe').click();
assert.match(document.getElementById('uiProbeStatus').textContent,/クリック正常（1回）/);
document.getElementById('uiProbe').click();
assert.match(document.getElementById('uiProbeStatus').textContent,/クリック正常（2回）/);
console.log('PASS: offline UI click diagnostics operate independently of pywebview');
// Exit before the intentionally unresolved bridge timeout; this test is about
// the local UI, not successful bridge startup.
process.exit(0);
