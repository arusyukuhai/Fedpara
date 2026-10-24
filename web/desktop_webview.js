'use strict';
const $ = id => document.getElementById(id);
let sources=[];
let codePath='';
function notify(message,failed=false){$('feedback').textContent=String(message);$('feedback').className=failed?'status-error':'status-ok';}
// Everything in this probe is local JavaScript; it NEVER calls Python.
let clickCount = 0;
function probeClick(){
  clickCount += 1;
  $('uiProbeStatus').textContent = `クリック正常（${clickCount}回）— 画面のJavaScriptは動作しています`;
  $('uiProbeStatus').className = 'status-ok';
}
function installDiagnostics(){
  $('uiProbe').addEventListener('click', probeClick);
  $('uiProbeStatus').textContent = 'JavaScript読み込み完了。クリックテストを押してください。';
  window.addEventListener('error',event => {
    const message = String(event.message || '不明なJavaScriptエラー');
    $('bridgeState').textContent = '画面エラー: '+ message;
    $('bridgeState').className = 'status-error';
  });
  window.addEventListener('unhandledrejection',event => {
    const message = String(event.reason?.message || event.reason || '不明な非同期エラー');
    $('bridgeState').textContent = '非同期エラー: '+ message;
    $('bridgeState').className = 'status-error';
  });
}
installDiagnostics();
function val(id){return $(id).value.trim();}
// Only fixed, explicit Python methods are callable from the embedded WebView.
// There is no fetch(), HTTP API, localhost listener, or shell interface.
const routes = Object.freeze({
  '/api/defaults': 'defaults',
  '/api/room/new': 'new_room',
  '/api/save': 'save_room',
  '/api/join': 'join_room',
  '/api/rooms': 'list_rooms',
  '/api/room/register': 'register_room',
  '/api/code': 'register_code',
  '/api/start': 'start_training',
  '/api/status': 'training_status',
  '/api/stop': 'stop_training',
  '/api/logs/copy': 'copy_logs',
  '/api/logs/save': 'save_logs',
  '/api/bench/start': 'start_benchmark',
  '/api/bench/status': 'benchmark_status'
});
// pywebview may expose the object before its dynamically generated functions.
// Wait for the actual methods, not just a truthy `window.pywebview`.
// Detect the minimum usable API rather than waiting forever for every method.
// Missing methods are reported only when invoked, so unrelated UI operations
// never become blocked by an optional bridge function.
const requiredBridgeMethods = Object.freeze(['defaults']);
function bridgeIsReady() {
  const exposed = window.pywebview?.api;
  return !!exposed && requiredBridgeMethods.every(name => typeof exposed[name] === 'function');
}

let bridgeWait = null;
function waitForBridge(timeoutMs=4500) {
  if (bridgeIsReady()) return Promise.resolve(window.pywebview.api);
  if (bridgeWait) return bridgeWait;
  bridgeWait = new Promise((resolve, reject) => {
    const started = Date.now();
    let timer = null;
    let finished = false;
    const cleanup = () => {
      if (finished) return;
      finished = true;
      window.removeEventListener('pywebviewready', check);
      if (timer !== null) clearTimeout(timer);
    };
    function check() {
      if (finished) return;
      if (bridgeIsReady()) {
        const ready = window.pywebview.api;
        cleanup();
        resolve(ready);
        return;
      }
      if (Date.now() - started >= timeoutMs) {
        const state = window.pywebview ? 'pywebview.api の公開が未完了' : 'pywebview 自体が未注入';
        cleanup();
        reject(new Error('Pythonブリッジを初期化できません (' + state +
          ')。この画面は通常のブラウザではなく `python -m patchlib.cli gui` で開いてください。' +
          ' pywebview を更新し、ターミナルのログを確認してください。'));
        return;
      }
      timer = setTimeout(check, 120);
    }
    window.addEventListener('pywebviewready', check);
    check();
  }).finally(() => { bridgeWait = null; });
  return bridgeWait;
}
async function api(path, payload={}) {
  const method = routes[path];
  if (!method) throw Error('許可されていない Python API: ' + path);
  const exposed = await waitForBridge();
  if (typeof exposed[method] !== 'function') throw Error('Python側にAPI '+method+' が見つかりません。最新版のPythonと画面ファイルを一緒に更新してください。');
  // A bridge method that hangs must not make the dashboard feel frozen.
  // The underlying Python call may still complete; this only times out UI wait.
  const timeoutMs = path === '/api/join' ? 12500 :
                    path === '/api/status' ? 3500 : 9500;
  let timer;
  try {
    return await Promise.race([
      Promise.resolve().then(() => exposed[method](payload)),
      new Promise((_,reject) => {
        timer=setTimeout(() => reject(new Error('Python応答が遅延しています。操作を再試行できます。')),timeoutMs);
      })
    ]);
  } finally {
    clearTimeout(timer);
  }
}
function renderSources(){
  const list=$('datasets');list.replaceChildren();
  if(!sources.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='データセットが未登録です。下のフォームから追加してください。';list.append(empty);return;}
  sources.forEach((s,i)=>{
    const box=document.createElement('div');box.className='dataset';
    const content=document.createElement('div');
    const title=document.createElement('strong');title.textContent=s.repo;
    const meta=document.createElement('small');meta.textContent=`${s.config||'default'} · ${s.split} · ${s.revision} · ${JSON.stringify(s.columns)} · weight ${s.weight}`;
    content.append(title,meta);
    const remove=document.createElement('button');remove.className='danger';remove.type='button';remove.textContent='削除';
    remove.addEventListener('click',()=>{sources.splice(i,1);renderSources();});
    box.append(content,remove);list.append(box);
  });
}
function currentSources(){return sources.map(s=>({...s,columns:{...s.columns}}));}
function payload(){return {room:val('room'),model:val('model'),revision:val('revision'),task:val('task'),ws:val('ws'),seed:Number(val('seed')),initialization:val('modelInitialization'),sources:currentSources(),code_path:codePath,code_url:val('codeUrl')};}
function errorHandler(label,fn){return async()=>{try{await fn();}catch(e){notify(label+': '+e.message,true);}};}
$('registerRoom').addEventListener('click',errorHandler('サーバールーム登録',async()=>{
  const config=payload();
  const saved=await api('/api/save',config);
  $('fingerprint').textContent='Manifest SHA256: '+saved.sha256;
  const result=await api('/api/room/register',{
    room:config.room,ws:config.ws,room_public:$('roomPublic').checked
  });
  notify(result.public ? '公開ルームをサーバーに登録しました。参加者へルームIDが公開されます。' : '非公開ルームをサーバーに登録しました。');
}));
// The directory is rendered from a snapshot obtained only on explicit refresh.
// Filtering / sorting happen in JS, without polling the signaling server.
let publicRooms = [];
let directoryFetchInFlight = false;
function applyRoomDefinition(info){
  const m=info.manifest;
  if (!m || !m.model || !m.datasets || !Array.isArray(m.datasets.sources)) throw Error('ルーム定義が不正です');
  $('model').value=m.model.id;$('revision').value=m.model.revision;$('task').value=m.model.architecture;
  $('modelInitialization').value=m.model.description==='fedpara:random-tinystories-1m:v1'?'random':'pretrained';
  $('seed').value=m.datasets.seed;$('codeUrl').value=m.code?.source||'';
  sources=m.datasets.sources;codePath='';renderSources();
  $('codeStatus').textContent=m.code?.sha256?`ルーム登録済みコード SHA256 ${m.code.sha256}（自動実行しません）`:'コード未登録';
  $('fingerprint').textContent='Manifest SHA256: '+info.sha256;
}
function roomLabel(room){return String(room||'').slice(0,10)+'…';}
function renderPublicRooms(){
  const list=publicRooms.slice();
  const query=val('roomFilter').toLocaleLowerCase();
  const availableOnly=$('roomAvailable').checked;
  const selected=list.filter(entry=>{
    const available=(Number(entry.capacity)||0)-(Number(entry.online)||0);
    if(availableOnly && available<=0)return false;
    const searchable=[entry.room_id,entry.model,...(Array.isArray(entry.datasets)?entry.datasets:[])].join(' ').toLocaleLowerCase();
    return !query || searchable.includes(query);
  });
  const sort=val('roomSort');
  selected.sort((a,b)=>{
    const aa=Number(a.online)||0,bb=Number(b.online)||0;
    if(sort==='model')return String(a.model||'').localeCompare(String(b.model||''))||String(a.room_id).localeCompare(String(b.room_id));
    if(sort==='available')return ((Number(b.capacity)||0)-bb)-((Number(a.capacity)||0)-aa)||String(a.room_id).localeCompare(String(b.room_id));
    return bb-aa || String(a.room_id).localeCompare(String(b.room_id));
  });
  $('roomDirectoryCount').textContent=`${selected.length}件表示 / 公開ルーム ${list.length}件`;
  const container=$('roomDirectory');container.replaceChildren();
  if(!selected.length){
    const empty=document.createElement('div');empty.className='empty';
    empty.textContent=list.length?'条件に一致するルームはありません。検索条件を変更してください。':'公開ルームはありません。更新ボタンで再取得してください。';
    container.append(empty);return;
  }
  for(const entry of selected){
    // Never interpret room metadata as HTML. The server response is untrusted.
    if(typeof entry.room_id!=='string'||typeof entry.model!=='string')continue;
    const card=document.createElement('article');card.className='room-card';
    const head=document.createElement('div');head.className='room-card-header';
    const title=document.createElement('strong');title.className='room-model';title.textContent=entry.model;
    const occupancy=document.createElement('span');
    const online=Math.max(0,Number(entry.online)||0),capacity=Math.max(0,Number(entry.capacity)||0);
    occupancy.className='room-occupancy'+(online>=capacity?' full':'');
    occupancy.textContent=`${online}/${capacity} 人`;
    head.append(title,occupancy);
    const data=document.createElement('div');data.className='room-subtitle';
    data.textContent='Datasets: '+(Array.isArray(entry.datasets)&&entry.datasets.length?entry.datasets.join(' · '):'未指定');
    const hint=document.createElement('div');hint.className='room-id-preview';hint.textContent='Room '+roomLabel(entry.room_id);
    const actions=document.createElement('div');actions.className='room-card-actions';
    const pick=document.createElement('button');pick.type='button';pick.className='secondary';pick.textContent='IDを選択';
    pick.addEventListener('click',()=>{$('room').value=entry.room_id;$('roomPublic').checked=true;notify('IDを選択しました。設定を読むには「設定を取得」を押してください。');});
    const load=document.createElement('button');load.type='button';load.textContent='設定を取得';
    load.addEventListener('click',errorHandler('ルーム読み込み',async()=>{
      load.disabled=true;
      try{
        const info=await api('/api/join',{room:entry.room_id,ws:val('ws')});
        applyRoomDefinition(info);
        $('room').value=entry.room_id;$('roomPublic').checked=true;
        notify('ルーム定義を読み込みました。署名指紋を作成者と照合してください。');
      }finally{load.disabled=false;}
    }));
    actions.append(pick,load);
    card.append(head,data,hint,actions);container.append(card);
  }
}
for (const id of ['roomFilter','roomSort','roomAvailable']) {
  $(id).addEventListener(id==='roomFilter'?'input':'change',renderPublicRooms);
}
$('refreshRooms').addEventListener('click',errorHandler('公開ルーム',async()=>{
  if(directoryFetchInFlight)return;
  directoryFetchInFlight=true;
  $('refreshRooms').disabled=true;
  $('roomDirectoryCount').textContent='サーバーからルーム一覧を取得しています…';
  try{
    const result=await api('/api/rooms',{ws:val('ws')});
    if(!Array.isArray(result.rooms))throw Error('公開ルーム一覧の形式が不正です');
    publicRooms=result.rooms.slice(0,200);
    renderPublicRooms();
  }catch(e){$('roomDirectoryCount').textContent='取得に失敗しました（最後の一覧を保持）';throw e;}
  finally{directoryFetchInFlight=false;$('refreshRooms').disabled=false;}
}));
$('newRoom').addEventListener('click',errorHandler('ルーム生成',async()=>{
  $('room').value=(await api('/api/room/new',{})).room;notify('新規ルームIDを生成しました');
}));
$('addDataset').addEventListener('click',errorHandler('データセット',async()=>{
  const cols={};for(const p of val('columns').split(',')){
    const pos=p.indexOf('=');if(pos<=0)throw Error('columns は text=text のように入力してください');
    cols[p.slice(0,pos).trim()]=p.slice(pos+1).trim();
  }
  if(sources.length>=12)throw Error('登録上限は12件です');
  sources.push({repo:val('repo'),revision:val('ds_revision'),config:val('config')||null,
                split:val('split'),columns:cols,weight:Number(val('weight'))});
  renderSources();notify('データセットを追加しました（保存前です）');
}));
$('importDatasets').addEventListener('change',errorHandler('インポート',async()=>{
  const file=$('importDatasets').files[0];if(!file)return;
  if(file.size>16*1024)throw Error('JSONは16KiB以下にしてください');
  const src=JSON.parse(await file.text());if(!Array.isArray(src)||src.length>12)throw Error('1〜12件の配列が必要です');
  sources=src;renderSources();notify('データセット定義を取り込みました');
}));
$('registerCode').addEventListener('click',errorHandler('コード登録',async()=>{
  const file=$('codeUpload').files[0];if(!file)throw Error('ファイルを選択してください');
  if(file.size>256*1024)throw Error('ファイル上限は256KiBです');
  const bytes=new Uint8Array(await file.arrayBuffer());let text='';
  for(let i=0;i<bytes.length;i+=0x4000)text+=String.fromCharCode(...bytes.slice(i,i+0x4000));
  const info=await api('/api/code',{name:file.name,contents:btoa(text)});
  codePath=info.path;
  $('codeStatus').textContent=`登録済み: ${info.kind} / SHA256 ${info.sha256}（自動実行しません）`;
  notify('コードの検証・ローカル登録が完了しました');
}));
$('save').addEventListener('click',errorHandler('保存',async()=>{
  const info=await api('/api/save',payload());
  $('fingerprint').textContent='Manifest SHA256: '+info.sha256;
  notify('保存: '+info.path);
}));
$('join').addEventListener('click',errorHandler('参加',async()=>{
  notify('ルームのデータセット定義を取得中…');
  const info=await api('/api/join',{room:val('room'),ws:val('ws')});
  applyRoomDefinition(info);
  notify('ルーム定義を読み込みました。作成者と指紋を別経路で照合してください。');
}));
$('start').addEventListener('click',errorHandler('学習開始',async()=>{
  const checkpoint=val('base');
  if(checkpoint && !checkpoint.toLowerCase().endsWith('.fstate'))throw Error('初期状態には .fstate ファイルを指定してください（空欄なら自動作成）。');
  notify('モデル・データセット設定を検証中…');
  const config=payload();
  const saved=await api('/api/save',config);
  $('fingerprint').textContent='Manifest SHA256: '+saved.sha256;
  const result=await api('/api/start',{room:config.room,base:val('base'),ws:config.ws,kind:val('kind'),rank:Number(val('rank')),reinit_every:Number(val('trainReinitEvery')),room_public:$('roomPublic').checked,optimizer:val('trainOptimizer'),initialization:val('modelInitialization'),
    lr:Number(val('trainLr')),local_steps:Number(val('trainLocalSteps')),max_length:Number(val('trainMaxLength'))});
  notify(result.auto_state ? '初期モデルを自動準備して学習開始: PID '+result.pid : '学習開始: PID '+result.pid);
  await poll();
}));
$('stop').addEventListener('click',errorHandler('学習停止',async()=>{
  await api('/api/stop',{});notify('学習プロセスへ停止指示を送信しました');
}));
$('copyLogs').addEventListener('click', errorHandler('ログをコピー', async()=>{
  const result=await api('/api/logs/copy');
  notify(`ログをクリップボードへコピーしました（${result.chars}文字）。ルームID等を共有する前に確認してください。`);
}));
$('saveLogs').addEventListener('click', errorHandler('ログを保存', async()=>{
  const result=await api('/api/logs/save');
  notify('ログを保存しました: '+result.path);
}));
$('selectLogs').addEventListener('click',()=>{
  const node=$('logs');
  const selection=window.getSelection?.();
  if (!selection || !document.createRange){
    notify('テキスト選択が使用できません。「ログをコピー」をお使いください。',true);return;
  }
  const range=document.createRange();range.selectNodeContents(node);
  selection.removeAllRanges();selection.addRange(range);
  node.focus?.();
  notify('ログを選択しました。⌘Cでコピーできます。');
});
let benchmarkInFlight = false;
function benchmarkMetric(label,value) {
  const card=document.createElement('div'); card.className='benchmark-stat';
  const l=document.createElement('small');l.textContent=label;
  const v=document.createElement('strong');v.textContent=value;
  card.append(l,v);return card;
}
async function pollBenchmark(){
  if (benchmarkInFlight || document.hidden) return;
  benchmarkInFlight=true;
  try {
    const state=await api('/api/bench/status');
    $('benchStatus').textContent=state.running ? '比較実験中…（PID '+(state.pid||'')+'）' :
       state.exit_code===null ? '未実行' :
       state.exit_code===0 ? '完了 · レポート保存済み' : '失敗 · 終了コード '+state.exit_code;
    if(state.logs) $('benchLog').textContent=state.logs;
    if(state.result) {
      const r=state.result, loss=r.loss, t=r.compute;
      const out=$('benchResults');out.replaceChildren();
      out.append(benchmarkMetric('中央集権 / Validation CE',Number(loss.central).toFixed(4)),
        benchmarkMetric('分散 / Validation CE',Number(loss.p2p).toFixed(4)),
        benchmarkMetric('中央集権 / CPU計算秒',Number(t.central_s).toFixed(3)+' s'),
        benchmarkMetric('分散 / CPU計算秒 ÷ ピア数',Number(t.p2p_divided_by_peers_s).toFixed(3)+' s'),
        benchmarkMetric('分散 / 逐次計算合計',Number(t.p2p_serial_s).toFixed(3)+' s'),
        benchmarkMetric('通信所要時間', '未計測'));
      const paths=Object.values(r.artifacts||{}).join(' / ');
      $('benchLog').textContent=(state.logs||'')+'\n保存先: '+paths;
    }
  }catch(e){$('benchStatus').textContent='ベンチマーク情報取得失敗: '+e.message;}
  finally {benchmarkInFlight=false;}
}
$('benchModel').addEventListener('change',()=>{
  if(val('benchModel')==='toy')$('benchInitialization').value='random';
  $('benchInitialization').disabled=val('benchModel')==='toy';
  $('benchDataset').disabled=val('benchModel')==='toy';
  $('benchSequences').value=val('benchModel')==='toy'?'64':'2048';
  $('benchValidationSequences').value=val('benchModel')==='toy'?'64':'1024';
});
$('trainOptimizer').addEventListener('change',()=>{
  if(val('trainOptimizer')==='lora-muon')$('kind').value='lora';
});
$('benchOptimizer').addEventListener('change',()=>{
  if(val('benchOptimizer')==='lora-muon') { $('benchKind').value='lora';$('benchLr').value='0.01'; }
});
$('benchStart').addEventListener('click',errorHandler('比較実験',async()=>{
  const options={peers:Number(val('benchPeers')),rounds:Number(val('benchRounds')),
    local_steps:Number(val('benchSteps')),reinit_every:Number(val('benchReinitEvery')),rank:Number(val('benchRank')),kind:val('benchKind'),
    optimizer:val('benchOptimizer'),
    lr:Number(val('benchLr')),
    model:val('benchModel'),initialization:val('benchInitialization'),dataset:val('benchModel')==='toy'?'auto':val('benchDataset'),
    sequences:Number(val('benchSequences')),
    validation_sequences:Number(val('benchValidationSequences'))};
  const started=await api('/api/bench/start',options);
  $('benchResults').replaceChildren();
  $('benchStatus').textContent='ベンチマークを起動しました（PID '+started.pid+'）';
  $('benchLog').textContent='出力先: '+started.output_dir;
  await pollBenchmark();
}));
let pollInFlight = false;
let lastLogText = null;
async function poll(){
  if (pollInFlight || document.hidden) return;
  pollInFlight = true;
  try {
    const result=await api('/api/status');
    $('runStatus').textContent=result.running?'学習中 · PID '+result.pid:'停止中'+(result.exit_code===null?'':' · 終了コード '+result.exit_code);
    const logText = result.logs || 'ログなし';
    if (lastLogText !== logText) {
      const logEl=$('logs');
      const atEnd = logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight < 40;
      // Do not destroy native text selection while a user is copying logs.
      const selection=window.getSelection?.();
      const selectingLogs=selection && !selection.isCollapsed &&
          (logEl.contains?.(selection.anchorNode) || logEl.contains?.(selection.focusNode));
      if (selectingLogs) return;
      logEl.textContent=logText;
      if (atEnd) logEl.scrollTop=logEl.scrollHeight;
      lastLogText=logText;
    }
  }catch(e){$('runStatus').textContent='GUIに接続できません: '+e.message;}
  finally {pollInFlight = false;}
}
let booted = false;
let booting = false;
let pollTimer = null;
async function boot(){
  if (booted || booting) return;
  booting = true;
  try {
    notify('Pythonブリッジの準備を確認中…');
    $('bridgeState').textContent='Pythonブリッジ接続中…（GUIのクリック操作は独立しています）';
    const state = await api('/api/defaults');
    for(const key of ['room','model','revision','task','ws','seed','kind','rank'])$(key).value=state[key];
    $('modelInitialization').value=state.initialization;
    $('trainReinitEvery').value=state.reinit_every;
    $('trainLr').value=state.lr;
    $('trainLocalSteps').value=state.local_steps;
    $('trainMaxLength').value=state.max_length;
    sources=state.sources;renderSources();
    booted = true;
    notify('準備完了 · Fedpara Desktop（WebView / Python）');
    $('bridgeState').textContent='Pythonブリッジ接続済み';
    $('bridgeState').className='status-ok';
    await poll();
    await pollBenchmark();
    if (pollTimer === null) pollTimer = setInterval(()=>{void poll();void pollBenchmark();},3500);
  } catch(e) {
    notify('起動失敗: ' + e.message,true);
    $('runStatus').textContent = 'Python API の準備ができていません';
    $('bridgeState').textContent = 'Pythonブリッジ未接続（クリックテストが動くなら画面自体は正常です）';
    $('bridgeState').className='status-error';
  } finally {
    booting = false;
  }
}
// The native event is the preferred signal, but an older or delayed renderer
// may inject the methods slightly before/after this event. A bounded poll in
// waitForBridge() also handles those cases and never spins synchronously.
window.addEventListener('pywebviewready', () => { if (!booted) void boot(); });
$('retryBridge').addEventListener('click', () => { void boot(); });
// When native WebKit resumes from sleep, promptly refresh once instead of
// accumulating repeated outstanding Python RPC requests.
if (typeof document.addEventListener === "function") document.addEventListener("visibilitychange",()=>{ if(!document.hidden)void poll(); });
void boot();
