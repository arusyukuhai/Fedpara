'use strict';
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict'), path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../web/desktop_webview.js'),'utf8');
const page=fs.readFileSync(path.join(__dirname,'../web/desktop_webview.html'),'utf8');
for(const id of ['roomFilter','roomSort','roomAvailable','roomDirectoryCount']) assert.match(page,new RegExp('id="'+id+'"'));
class Element {
  constructor(id){this.id=id;this.textContent='';this.value='';this.checked=false;this.disabled=false;this.files=[];this.style={};this.children=[];this.handlers={};}
  addEventListener(event, cb){this.handlers[event]=cb;}
  replaceChildren(){this.children=[];}
  append(...children){this.children.push(...children);}
  async trigger(event='click'){return this.handlers[event]?.();}
}
const elements=new Map();const document={hidden:false,getElementById(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);},createElement(name){return new Element(name);},addEventListener(){}};
const $=id=>document.getElementById(id);
const roomA={room_id:'aaa-123',model:'foo/model',datasets:['alpha/text'],online:2,capacity:8};
const roomB={room_id:'bbb-345',model:'bar/model',datasets:['beta/corpus'],online:8,capacity:8};
const roomC={room_id:'ccc-678',model:'other/model',datasets:['alpha/code'],online:3,capacity:16};
let reads=0, joins=0;
const mockAPI={defaults:async()=>({room:'room-test',model:'unit/test',revision:'main',task:'causal-lm',ws:'ws://127.0.0.1:8080/ws',seed:42,kind:'loha',rank:8,sources:[]}),training_status:async()=>({running:false,logs:''}),list_rooms:async()=>{reads++;return {rooms:[roomA,roomB,roomC]}},join_room:async()=>{joins++;return {manifest:{model:{id:'bar/model',revision:'main',architecture:'causal-lm'},datasets:{seed:11,sources:[]},code:{source:'',sha256:''}},sha256:'h'.repeat(64)}}};
const window={pywebview:{api:mockAPI},addEventListener(){},removeEventListener(){}};
const context=vm.createContext({window,document,Date,Promise,Number,String,Error,setTimeout,clearTimeout,setInterval:()=>1,Uint8Array});
vm.runInContext(code,context);
async function run(){
  await new Promise(r=>setTimeout(r,20));
  await $('refreshRooms').trigger();
  assert.equal(reads,1);assert.equal($('roomDirectory').children.length,3);
  $('roomFilter').value='beta';await $('roomFilter').trigger('input');
  assert.equal(reads,1,'filter must not request the server again');
  assert.equal($('roomDirectory').children.length,1);
  $('roomFilter').value='';$('roomAvailable').checked=true;await $('roomAvailable').trigger('change');
  assert.equal($('roomDirectory').children.length,2);
  const card=$('roomDirectory').children[0];
  assert.equal(card.children.length,4);
  const choose=card.children[3].children[0];await choose.trigger();
  assert.equal($('room').value,'ccc-678');
  const load=card.children[3].children[1];await load.trigger();
  assert.equal(joins,1);assert.equal($('model').value,'bar/model');
  console.log('PASS room search/sort/filter/quick-load no extra server round trips');
}
run().catch(e=>{console.error(e);process.exitCode=1});
