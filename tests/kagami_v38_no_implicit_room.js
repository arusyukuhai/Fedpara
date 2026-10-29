'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/desktop_webview.js'), 'utf8');
const nodes = new Map();
class Node {
  constructor(id) {
    this.id=id;this.value='';this.textContent='';this.innerHTML='';this.title='';
    this.checked=false;this.disabled=false;this.files=[];this.children=[];this.dataset={};
    this.style={};this.handlers={};this.classList={add(){},remove(){},toggle(){}};
  }
  addEventListener(name, cb) {this.handlers[name]=cb;}
  async fire(name='click'){return this.handlers[name]?.();}
  setAttribute(){}
  replaceChildren(...children){this.children=children;}
  append(...children){this.children.push(...children);}
  querySelectorAll(){return [];}
}
const document={hidden:false,documentElement:{lang:'ja'},
  getElementById(id){if(!nodes.has(id))nodes.set(id,new Node(id));return nodes.get(id);},
  createElement(tag){return new Node(tag);},createElementNS(_ns,tag){return new Node(tag);},
  querySelectorAll(){return [];},addEventListener(){}};
const $=id=>document.getElementById(id);
const rpcCalls=[];
const sharedSettings={room:'R'.repeat(32), model:'org/model',revision:'main',task:'causal-lm',
  ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',rank:16,initialization:'pretrained',
  reinit_every:20,lr:.0001,disk_cache_fraction:.6,wide_lora_rank:4,push_fanout:2,max_journal_gb:20,
  sources:[{repo:'org/corpus',revision:'main',config:null,split:'train',columns:{text:'text'},weight:1}]};
const api=new Proxy({}, {get(_target,name){if(name==='then')return undefined;return async arg=>{
  rpcCalls.push([name,arg]);
  if(name==='defaults')return sharedSettings;
  if(name==='training_status')return {running:false,model_id:null,telemetry:[],logs:''};
  if(name==='device_capabilities')return {device:'cpu',ram_bytes:8e9,vram_bytes:0};
  if(name==='local_models')return {models:[{room_id:'S'.repeat(32),model:'org/model',revision:'main',datasets:['org/corpus'],task:'causal-lm',local:true}]};
  if(name==='list_rooms')return {rooms:[]};
  if(name==='recommend_training')return {rank:16,rank_pinned:false,can_train:true,max_length:256,local_steps:12};
  if(name==='available_checkpoints')return {checkpoints:[]};
  if(name==='eval_status')return {running:false};
  return {};
};}});
const window={pywebview:{api},addEventListener(){},removeEventListener(){}};
const context=vm.createContext({document,window,setInterval:()=>1,setTimeout,clearTimeout,
  Number,String,Error,Date,Promise,Math,JSON,URL,Uint8Array});
vm.runInContext(source,context);
(async()=>{
  await new Promise(resolve=>setTimeout(resolve,100));
  assert.equal($('room').value,sharedSettings.room,'boot creates an unsaved draft ID');
  await $('start').fire();
  await new Promise(resolve=>setTimeout(resolve,30));
  assert.match($('feedback').textContent,/MODELS/);
  assert.equal(rpcCalls.filter(x=>x[0]==='save_room').length,0);
  assert.equal(rpcCalls.filter(x=>x[0]==='start_training').length,0);
  console.log('PASS: starting training with auto-generated draft cannot fork a room');
})().catch(err=>{console.error(err);process.exitCode=1});
