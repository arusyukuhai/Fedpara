'use strict';
// Kagami: local WebView controller. Fixed Python RPC allowlist; no HTTP listener.
const $ = (id) => document.getElementById(id);
const translations = {
  ja: {
    workspace:'ワークスペース',system:'システム',checkpoints:'チェックポイント',weightFiles:'モデル重み',genesis:'初期重み',
    modelsSubtitle:'分散モデルを見つけて管理する',language:'言語',connecting:'接続中',
    modelCurrentLoss:'直近のLoss',modelCurrentAccuracy:'直近のTop-1',autoRankLabel:'ランク（自動・共通）',
    modelsIntro:'モデルはアーキテクチャと既定データセットを含む学習空間です。',refreshModels:'一覧を更新',
    source:'表示',all:'すべて',local:'ローカル',public:'公開',searchModels:'モデル名・データセットで検索',
    noModels:'モデルがまだありません。「一覧を更新」を押してください。',independentExperiment:'同じ設定でも別の実験として作成する',selectedModel:'選択中のモデル',
    networkPeers:'オンライン人数',architecture:'アーキテクチャ',sizeEstimate:'規模の目安',revision:'リビジョン',
    defaultDatasets:'既定のデータセット',useModel:'このモデルを選択',goTraining:'学習画面へ',
    trainingIntro:'LoHa / LoRA · 継続FedSum · ローカル学習メトリクス',start:'▶ 開始 / 再開',
    pause:'Ⅱ 学習停止（中継継続）',disconnect:'ネットワークから切断',saveLog:'ログを保存',localLoss:'ローカルミニバッチ',
    accuracy:'トークン正解率',localAccuracy:'学習中のTop-1（検証ではない）',throughput:'学習速度',
    historyWindow:'表示期間',chartBoth:'両方',chartLossOnly:'Lossを拡大',chartAccuracyOnly:'Top-1を拡大',entireRun:'実行全期間',lossChart:'Lossの推移',accuracyChart:'Top-1正解率の推移',activity:'直近ログ',copy:'コピー',
    trainingSettings:'学習設定',adapter:'アダプター',optimizer:'最適化手法',learningRate:'学習率',
    localSteps:'公開ごとのステップ',contextLength:'文脈長',peers:'参加者（観測値）',cacheFraction:'VRAMキャッシュ率',
    streaming:'重み読込',normal:'通常',streamMode:'ストリーミング',reinitEvery:'再初期化間隔',
    advanced:'詳細設定',wideRank:'Embedding rank',journalSize:'履歴上限 GiB',baseCheckpoint:'初期 .fstate（任意）',
    settingsNote:'設定は新規ワーカー起動時に適用。学習停止中も署名済み更新とモデル断片を中継します。',
    datasetsIntro:'公開データセットを検索し、新規モデルに複数追加できます。',search:'検索',
    searchHF:'Hugging Faceでデータセットを探す',searchToStart:'公開Hugging Face Hubを検索してください。',
    hubCatalog:'Hugging Face 公開カタログ',attachedDatasets:'モデルに追加するデータセット',datasetRepo:'リポジトリ',
    weight:'混合ウェイト',columnMap:'列マッピング（例 text=text）',addDataset:'＋ データセットを追加',
    evalIntro:'学習済み重みから実際に推論します。',trainedEval:'学習済みモデル',evalScope:'使用する重み',captureSnapshot:'現在の重みを保存',baseOnly:'初期重み（手動選択）',currentModel:'モデル',
    prompt:'プロンプト',precision:'精度 / 量子化',maxTokens:'生成最大トークン',generate:'▶ 推論',cancel:'中止',
    evalDisclaimer:'実効重みのチェックポイントを検証して読み込みます。量子化には一時ディスクが必要になる場合があります。',
    output:'出力',networkIntro:'この端末から現在確認できるWebRTC接続だけを可視化します。',
    connections:'現在の接続',localView:'自端末から見える隣接グラフ（全体図ではありません）',
    networkTelemetry:'ネットワーク統計',connectedPeers:'接続中ピア',duplicates:'重複イベント',
    appliedEvents:'適用した更新',inbox:'受信キュー',
    networkNote:'直接接続した隣接ピアのみ表示します。他ピア間の接続は観測できないため描きません。',
    newIntro:'アーキテクチャ、データセット集合、実行設定を固定して新規モデルを作成します。',
    newId:'新しいID',createModel:'モデルを作成',identity:'モデルの識別情報',
    modelId:'Hugging FaceモデルID',modelRevision:'リビジョン（コミットSHA推奨）',
    creationMode:'作成方式',tokenizerModel:'使用するTokenizer',tokenizerRevision:'Tokenizerリビジョン',copyWeights:'既存モデルを重みごとコピー',copyArchitecture:'既存モデルの構造だけコピー',codeArchitecture:'Pythonで構造を設計（Transformers Config）',configInit:'構造から再初期化',initialization:'初期化',randomInit:'TinyStoriesランダム初期化',pretrained:'事前学習済み',task:'タスク',
    visibility:'公開範囲',private:'非公開',signaling:'シグナリング WebSocket',networkId:'ネットワーク識別子',
    pickDatasets:'データセットを探す',modelCode:'モデルアーキテクチャ・Python',notExecuted:'安全な構造解析',
    codeNote:'安全なbuild_config()を静的解析して参加ピアで再構成します。任意Python文の実行はできません。',
    loadCode:'.pyを開く',codeSource:'コード配布元URL（任意）',publishModel:'選択中モデルを公開登録',
    viewTraining:'学習を表示',download:'選択',notChecked:'未検証',cannotTrain:'学習不可',candidate:'要動作確認',
    needSelect:'モデルを選択してください',offline:'停止中',relay:'中継のみ',training:'学習中',
    stopping:'終了処理中',idle:'停止中',unknown:'判定できません',modelSaved:'モデル設定を保存しました',
    alreadyTraining:'学習は既に動作しています',trainStarted:'学習を開始しました',trainResumed:'学習を再開しました',
    trainPaused:'学習を停止しました。更新とモデルの中継は続いています。',
    disconnecting:'切断を要求しました。チェックポイント保存を待ってください。',
    noAccuracy:'タスクが非対応、または正解率の計測待ち',unknownModel:'対応可否は未確認です。',
    autoTraining:'自動最適化 · 動的推定',autoNote:'モデル・メモリから自動推定。公開間隔はオンラインピア数で再調整されます。',datasetDuplicate:'同じデータセットは既に追加されています',datasetAdded:'データセットを追加しました',datasetLimit:'一つのモデルにつき最大12件です',
    pleaseSearch:'データセットを検索してください',modelSelected:'モデル設定を読み込みました',
    hubError:'Hubへの接続に失敗しました',evalRunning:'推論プロセスでモデルを読み込んでいます',
    noBrowser:'Pythonブリッジがありません。python -m patchlib.cli gui で起動してください。',
    noNetwork:'学習ピアが接続されていません',lastSeen:'直近メトリクスから取得',checkNeeded:'環境依存・未検証',
    unsupported:'未対応',createFirst:'最初にモデルを作成してください',noDataset:'データセットを追加してください',
    localScope:'ローカル',remoteScope:'公開',current:'選択中'
  },
  en: {
    workspace:'WORKSPACE',system:'SYSTEM',checkpoints:'Checkpoints',weightFiles:'Weights',genesis:'Genesis',
    modelsSubtitle:'Discover and manage distributed models',language:'Language',connecting:'Connecting',
    modelCurrentLoss:'LATEST LOSS',modelCurrentAccuracy:'LATEST TOP-1',autoRankLabel:'Rank · auto / shared',
    modelsIntro:'Each model includes architecture, default datasets, and a shared training space.',refreshModels:'Refresh models',
    source:'Source',all:'All',local:'Local',public:'Public',searchModels:'Search models or datasets',
    noModels:'No models yet. Refresh the directory to begin.',independentExperiment:'Create an independent experiment with identical settings',selectedModel:'SELECTED MODEL',
    networkPeers:'ONLINE PEERS',architecture:'ARCHITECTURE',sizeEstimate:'SIZE ESTIMATE',revision:'REVISION',
    defaultDatasets:'DEFAULT DATASETS',useModel:'Select this model',goTraining:'Open training',
    trainingIntro:'LoHa / LoRA · continual FedSum · local training metrics',start:'▶ Start / resume',
    pause:'Ⅱ Pause · keep relaying',disconnect:'Disconnect',saveLog:'Save log',localLoss:'Local minibatch',
    accuracy:'TOKEN ACCURACY',localAccuracy:'Local top-1, not validation',throughput:'Throughput',
    historyWindow:'DISPLAY RANGE',chartBoth:'Both',chartLossOnly:'Enlarge loss',chartAccuracyOnly:'Enlarge Top-1',entireRun:'実行全期間',lossChart:'Loss history',accuracyChart:'Top-1 accuracy',activity:'RECENT ACTIVITY',copy:'Copy',
    trainingSettings:'TRAINING SETTINGS',adapter:'Adapter',optimizer:'Optimizer',learningRate:'Learning rate',
    localSteps:'Steps / publication',contextLength:'Context length',peers:'Online peers (observed)',cacheFraction:'VRAM cache fraction',
    streaming:'Weight loading',normal:'Normal',streamMode:'Streamed',reinitEvery:'Reinitialization interval',
    advanced:'Advanced options',wideRank:'Embedding rank',journalSize:'Journal limit GiB',baseCheckpoint:'Initial .fstate (optional)',
    settingsNote:'Settings apply on worker startup. Signed updates and snapshots keep relaying while training is paused.',
    datasetsIntro:'Search the public Hub and attach multiple datasets to a new model.',search:'Search',
    searchHF:'Search Hugging Face datasets',searchToStart:'Search the public Hugging Face Hub to begin.',
    hubCatalog:'Hugging Face catalog',attachedDatasets:'DATASETS ATTACHED TO DRAFT',datasetRepo:'Repository',
    weight:'Mix weight',columnMap:'Column mapping (e.g. text=text)',addDataset:'＋ Add dataset',
    evalIntro:'Generate from the actual trained Kagami checkpoint.',trainedEval:'TRAINED CHECKPOINT',evalScope:'WEIGHTS SOURCE',captureSnapshot:'Save current weights',baseOnly:'Original HF base (explicit)',currentModel:'CURRENT MODEL',
    prompt:'PROMPT',precision:'PRECISION / QUANTIZATION',maxTokens:'MAX NEW TOKENS',generate:'▶ Generate',cancel:'Cancel',
    evalDisclaimer:'Verified effective-weight shards are loaded. Quantization may need substantial temporary disk space.',
    output:'OUTPUT',networkIntro:'Visualize real WebRTC links observable from this local peer.',
    connections:'LIVE CONNECTIONS',localView:'Local peer view (not global topology)',
    networkTelemetry:'NETWORK TELEMETRY',connectedPeers:'Connected peers',duplicates:'Duplicate events',
    appliedEvents:'Applied updates',inbox:'Inbound queue',
    networkNote:'Only directly observed links are drawn. The links between other peers are not observable.',
    newIntro:'Create a model with immutable architecture, dataset collection, and training config.',
    newId:'New ID',createModel:'Create model',identity:'MODEL IDENTITY',
    modelId:'Hugging Face model ID',modelRevision:'Revision (commit SHA recommended)',
    creationMode:'CREATE FROM',tokenizerModel:'Tokenizer (HF repository)',tokenizerRevision:'Tokenizer revision',copyWeights:'Copy pretrained model and weights',copyArchitecture:'Copy architecture, initialize new weights',codeArchitecture:'Build architecture in Python (HF Config)',configInit:'Seeded configuration',initialization:'Initialization',randomInit:'Random TinyStories initialization',pretrained:'Pretrained',task:'Task',
    visibility:'Visibility',private:'Private',signaling:'Signaling WebSocket',networkId:'Network identifier',
    pickDatasets:'Choose datasets',modelCode:'MODEL ARCHITECTURE · PYTHON',notExecuted:'STATIC CONFIG',
    codeNote:'Statically parsed build_config() can reconstruct built-in Transformers configs on peers. Arbitrary Python cannot run.',
    loadCode:'Open .py',codeSource:'Pinned source URL (optional)',publishModel:'Publish selected model',
    viewTraining:'View training',download:'Select',notChecked:'Unverified',cannotTrain:'Unavailable',candidate:'Needs test',
    needSelect:'Select a model first',offline:'Offline',relay:'Relay only',training:'Training',
    stopping:'Shutting down',idle:'Idle',unknown:'Unknown',modelSaved:'Model manifest saved',
    alreadyTraining:'Training is already running',trainStarted:'Training started',trainResumed:'Training resumed',
    trainPaused:'Training paused. Gossip and snapshot relay continue.',
    disconnecting:'Disconnect requested. Wait for checkpoint completion.',
    noAccuracy:'Waiting for accuracy or task unsupported',unknownModel:'Compatibility is not confirmed.',
    autoTraining:'AUTO TUNING · LIVE',autoNote:'Estimated from model/device; publish interval adapts to online membership.',datasetDuplicate:'This dataset is already attached',datasetAdded:'Dataset attached',datasetLimit:'A model supports up to 12 datasets',
    pleaseSearch:'Search for a dataset',modelSelected:'Model settings loaded',
    hubError:'Cannot reach Hugging Face Hub',evalRunning:'Inference worker is loading model weights',
    noBrowser:'Python bridge missing. Launch python -m patchlib.cli gui.',
    noNetwork:'No training peer is connected',lastSeen:'From latest telemetry',checkNeeded:'Device-dependent / unverified',
    unsupported:'Unsupported',createFirst:'Create a model to begin',noDataset:'Add a dataset first',
    localScope:'Local',remoteScope:'Public',current:'Selected'
  }
};
let lang='ja';let currentTab='models';let sources=[];let codePath='';let models=[];
let selected=null;let activatedRoom=null;let selectedByUser=false;let modelPage=0;let datasetsPage=0;let hubResults=[];let run={running:false};
let capabilities={};let latestTelemetry=[];let chartTelemetry=[];let lastHistoryFetch=0;let historyPending=false;let historyVersion=0;let startup=false;let lastLog='';let autoPlan=null;let autoPlanRequest=0;
const compatibilityPending=new Set();let compatibilityActive=0;const compatibilityQueue=[];
const t=(key)=>translations[lang][key]||key;
const m=(x)=>Number.isFinite(Number(x))?Number(x):0;
const short=(s,n=18)=>String(s||'—').length>n?String(s).slice(0,n)+'…':String(s||'—');
function translate(){document.documentElement.lang=lang;document.querySelectorAll('[data-t]').forEach(n=>{n.textContent=t(n.dataset.t)});document.querySelectorAll('[data-ph]').forEach(n=>n.placeholder=t(n.dataset.ph));$('screenSubtitle').textContent=t(currentTab==='models'?'modelsSubtitle':currentTab==='training'?'trainingIntro':currentTab==='datasets'?'datasetsIntro':currentTab==='eval'?'evalIntro':currentTab==='network'?'networkIntro':'newIntro');drawModels();drawSources();drawHub();drawSelection();drawRun();drawGraph();}
let toastTimer=null;
function notify(msg,error=false){const node=$('feedback');node.textContent=String(msg);node.classList.toggle('error',error);node.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.remove('show'),6500);}
function navigate(id){currentTab=id;for(const b of document.querySelectorAll('[data-tab]')){const active=b.dataset.tab===id;b.classList.toggle('current',active);b.setAttribute('aria-selected',active?'true':'false');}for(const panel of document.querySelectorAll('.screen'))panel.classList.toggle('active',panel.id==='panel-'+id);if(id==='training'&&run.model_id)void refreshChartHistory().catch(()=>{});if(id==='datasets'&&!hubResults.length)void searchDatasets().catch(e=>{setText('datasetCount','Hub unavailable');$('datasetInspectNote').textContent=String(e.message||e);});$('breadcrumb').textContent='KAGAMI / '+(id==='new'?'NEW MODEL':id.toUpperCase());$('screenSubtitle').textContent=t(id==='models'?'modelsSubtitle':id==='training'?'trainingIntro':id==='datasets'?'datasetsIntro':id==='eval'?'evalIntro':id==='network'?'networkIntro':'newIntro');}
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>navigate(b.dataset.tab)));
$('language').addEventListener('change',()=>{lang=$('language').value;translate();updateWsNotice();});
function get(id){return String($(id).value||'').trim();}
function updateWsNotice(){
  let msg='';const url=get('ws');
  try{const parsed=new URL(url);const local=['localhost','127.0.0.1','[::1]'].includes(parsed.hostname);
    if(local){msg=lang==='ja'?'ローカル接続専用です。インターネット越しには公開WSSシグナリングとTURN設定が必要です。':'Local-only URL. Public networks need a reachable WSS signaling server and TURN setup.';}
    else if(parsed.protocol!=='wss:'){msg=lang==='ja'?'リモート接続では暗号化されたwss://が必要です。':'Remote signaling must use encrypted wss://.';}
  }catch(_){msg=lang==='ja'?'WebSocketのURLを確認してください。':'Check WebSocket URL.';}
  setText('wsNotice',msg);
}
$('ws').addEventListener('input',updateWsNotice);
function get(id){return String($(id).value||'').trim();}
const routes=Object.freeze({defaults:'defaults',newRoom:'new_room',save:'save_room',join:'join_room',register:'register_room',
  publicModels:'list_rooms',localModels:'local_models',select:'select_model',compat:'model_compatibility',
  mixedSettings:'mixed_settings',pickMixFile:'pick_training_file',
  mixPinDataset:'pin_mixed_dataset',hubDiscover:'discover_hub_assets',rewardSearch:'search_reward_models',rewardInspect:'inspect_reward_model',device:'device_capabilities',datasets:'search_datasets',inspect:'inspect_dataset',plan:'recommend_training',code:'register_code',start:'start_training',
  status:'training_status',history:'training_history',pause:'stop_training',resume:'resume_training',disconnect:'disconnect_training',
  open:'open_storage',paths:'storage_paths',copy:'copy_logs',saveLog:'save_logs',eval:'evaluate',evalStatus:'eval_status',cancelEval:'cancel_eval',evalCheckpoints:'available_checkpoints',captureSnapshot:'request_eval_snapshot',snapshotStatus:'eval_snapshot_status'});
let bridgeWait;
function ready(){return !!window.pywebview?.api&&typeof window.pywebview.api.defaults==='function';}
async function waitBridge(){if(ready())return window.pywebview.api;if(bridgeWait)return bridgeWait;bridgeWait=new Promise((resolve,reject)=>{const started=Date.now();let timer=null;function check(){if(ready()){clearTimeout(timer);resolve(window.pywebview.api);return;}if(Date.now()-started>9000){reject(new Error(t('noBrowser')));return;}timer=setTimeout(check,110);}check();}).finally(()=>{bridgeWait=null;});return bridgeWait;}
async function api(action,payload={}){const name=routes[action];if(!name)throw Error('Unknown RPC route');const bridge=await waitBridge();if(typeof bridge[name]!=='function')throw Error('Missing Python RPC: '+name);
  const ms=(action==='hubDiscover'||action==='datasets'||action==='inspect'||action==='plan'||action==='publicModels'||action==='join'||action==='select'||action==='rewardSearch'||action==='rewardInspect'||action==='mixPinDataset')?16500:(action==='eval'||action==='captureSnapshot')?45000:action==='status'?4500:10000;
  let timer;try{return await Promise.race([Promise.resolve().then(()=>bridge[name](payload)),new Promise((_,reject)=>timer=setTimeout(()=>reject(Error('RPC timeout: '+action)),ms))]);}finally{clearTimeout(timer);}}
const safe=(title,call)=>async (...args)=>{try{return await call(...args);}catch(e){notify(title+': '+String(e.message||e),true);return null;}};
function create(tag,cls,text){const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=String(text);return el;}
function setText(id,text){$(id).textContent=String(text);}
function taglist(container,arr){container.replaceChildren();if(!arr||!arr.length){container.append(create('span','muted','—'));return;}for(const v of arr.slice(0,12))container.append(create('span','tag',v));}
function setPageControls(prefix,page,pages,count){setText(prefix+'Page',(page+1)+' / '+pages);$(prefix+'Prev').disabled=page<=0;$(prefix+'Next').disabled=page>=pages-1;if(prefix==='model')setText('modelCount',count+' models');}
function modelKey(x){return x.room_id;}
function modelFiltered(){const q=get('modelFilter').toLowerCase();const scope=get('modelScope');return models.filter(x=>(scope==='all'||(scope==='local'?x.local:!x.local))&&[x.model,...(x.datasets||[]),x.room_id].join(' ').toLowerCase().includes(q));}
function statusClass(x){return x==='unavailable'||x==='unsupported'?'red':x==='candidate'||x==='unknown'?'amber':'gray';}
function compatName(comp){if(!comp)return t('notChecked');return comp.status==='unavailable'||comp.status==='unsupported'?t('cannotTrain'):comp.status==='candidate'?t('candidate'):t('unknown');}
function drawModels(){const list=modelFiltered();const pages=Math.max(1,Math.ceil(list.length/4));modelPage=Math.min(modelPage,pages-1);const root=$('modelCards');root.replaceChildren();if(!list.length){root.append(create('div','empty',t('noModels')));}for(const item of list.slice(modelPage*4,(modelPage+1)*4)){
    const card=create('button','model-card'+(selected&&selected.room_id===item.room_id?' picked':''));card.type='button';const head=create('div','model-card-head');head.append(create('strong','',short(item.model.split('/').pop(),28)));
    const comp=item.compatibility;head.append(create('span','pill '+statusClass(comp?.status),compatName(comp)));card.append(head);
    card.append(create('div','mono muted truncate',item.model));
    const related=models.filter(x=>x.model===item.model&&x.revision===item.revision).length;
    const idText='ID '+String(item.room_id).slice(0,12)+' · '+(item.local?t('localScope'):t('remoteScope'))+' · '+m(item.online)+' '+t('networkPeers')+(related>1?' · '+related+' distinct IDs':'');
    const idNode=create('small','',idText);idNode.title=String(item.room_id);card.append(idNode);
    const tags=create('div','tag-list');taglist(tags,item.datasets);card.append(tags);card.addEventListener('click',()=>{selected=item;selectedByUser=true;drawModels();drawSelection();void checkCompatibility(item);});root.append(card);
  }setPageControls('model',modelPage,pages,list.length);setText('modelsCounter',models.length);
  // Probe only the visible page, without blocking rendering or flooding the Hub.
  if(ready())for(const item of list.slice(modelPage*4,(modelPage+1)*4))queueCompatibility(item);}
function drawSelection(){const x=selected;setText('selectedModelTitle',x?x.model.split('/').pop(): '—');setText('selectedModelId',x?x.model:'—');setText('modelPeerCount',x?m(x.online):'—');setText('modelTask',x?.task||'causal-lm');setText('modelRevision',x?short(x.revision||'—',15):'—');setText('modelSize',x?.compatibility?.parameters?formatParams(x.compatibility.parameters):'—');taglist($('modelDatasets'),x?.datasets||[]);
  // Telemetry belongs to the active room, never to an arbitrary selected card.
  const sample=x?.room_id===run.model_id?latestTelemetry.at(-1):null;
  const lastLoss=sample?.loss,lastAccuracy=sample?.accuracy;
  setText('modelCurrentLoss',lastLoss!=null&&Number.isFinite(Number(lastLoss))?Number(lastLoss).toFixed(4):'—');
  setText('modelCurrentAccuracy',lastAccuracy!=null&&Number.isFinite(Number(lastAccuracy))?(100*Number(lastAccuracy)).toFixed(2)+'%':'—');
  const cap=x?.compatibility;const badge=$('modelCompatibility');badge.textContent=compatName(cap);badge.className='pill '+statusClass(cap?.status);setText('compatibilityReason',cap?.reason||t('unknownModel'));$('selectModel').disabled=!x||cap?.status==='unsupported'||cap?.status==='unavailable';$('modelTrain').disabled=!x||cap?.status==='unsupported'||cap?.status==='unavailable';}
function formatParams(v){return v>=1e9?(v/1e9).toFixed(1)+'B':v>=1e6?(v/1e6).toFixed(0)+'M':String(v);}
function queueCompatibility(item){
  if(!item||item.compatibility||compatibilityPending.has(item.room_id))return;
  compatibilityPending.add(item.room_id);compatibilityQueue.push(item);drainCompatibility();
}
function drainCompatibility(){
  while(compatibilityActive<2&&compatibilityQueue.length){
    const item=compatibilityQueue.shift();compatibilityActive++;
    void (async()=>{
      const result=await api('compat',{model:item.model,revision:item.revision||'main'}).catch(()=>({status:'unknown',reason:'Hub metadata not available',parameters:null}));
      const found=models.find(x=>x.room_id===item.room_id);
      if(found){found.compatibility=result;drawModels();if(selected?.room_id===item.room_id){selected=found;drawSelection();}}
    })().finally(()=>{compatibilityActive--;compatibilityPending.delete(item.room_id);drainCompatibility();});
  }
}
function checkCompatibility(item){queueCompatibility(item);}
async function refreshModels(){const local=await api('localModels');const remote=await api('publicModels',{ws:get('ws')}).catch(e=>{notify('Public directory: '+e.message,true);return {rooms:[]};});const map=new Map();for(const x of remote.rooms||[])map.set(x.room_id,{...x,local:false});for(const x of local.models||[]){const prev=map.get(x.room_id);map.set(x.room_id,{...prev,...x,online:prev?.online||0,local:true});}
  models=Array.from(map.values()).slice(0,250).sort((a,b)=>Number(b.is_training||false)-Number(a.is_training||false)||Number(b.online)-Number(a.online));
  if(selected){selected=models.find(x=>modelKey(x)===modelKey(selected))||null;}if(!selected&&models.length)selected=models[0];drawModels();drawSelection();if(selected)void checkCompatibility(selected);}
$('refreshModels').addEventListener('click',safe('MODELS',refreshModels));$('modelFilter').addEventListener('input',()=>{modelPage=0;drawModels();});$('modelScope').addEventListener('change',()=>{modelPage=0;drawModels();});
$('modelPrev').addEventListener('click',()=>{modelPage=Math.max(0,modelPage-1);drawModels();});$('modelNext').addEventListener('click',()=>{modelPage++;drawModels();});
function applyManifest(info,room){const v=info.manifest;if(!v?.model||!Array.isArray(v.datasets?.sources))throw Error('Invalid manifest');$('room').value=room;$('model').value=v.model.id;$('revision').value=v.model.revision;$('task').value=v.model.architecture;if(v.adapter?.kind)$('kind').value=v.adapter.kind;$('adapterScope').value=v.adapter?.target_scope||'auto';$('creationMode').value=v.design?.mode==='code'?'code':v.design?.mode==='architecture'?'architecture':'weights';$('modelInitialization').value=v.design?'config':v.model.description==='fedpara:random-tinystories-1m:v1'?'random':'pretrained';$('base').value='';syncCreationMode(false);$('seed').value=v.datasets.seed;$('codeUrl').value=v.code?.source||'';$('tokenizerModel').value=v.design?.tokenizer?.id||v.model.id;$('tokenizerRevision').value=v.design?.tokenizer?.revision||v.model.revision;sources=v.datasets.sources.map(s=>({...s,columns:{...s.columns}}));codePath='';$('codeEditor').value='';setText('codeStatus',v.code?.sha256?'SHA-256 '+v.code.sha256:'—');setText('fingerprint','SHA-256 '+info.sha256);drawSources();void refreshAutoPlan();}
async function selectModel(){
  if(!selected)throw Error(t('needSelect'));
  // Allow selecting existing models even when this device can only relay.
  // A later training attempt can independently report device incompatibility.
  activatedRoom=null;
  const result=await api('select',{room:selected.room_id,ws:get('ws')});
  applyManifest(result,selected.room_id);await reloadMixed(selected.room_id);
  if(result.local===false){
    // Selecting a PUBLIC model is an explicit JOIN of that existing room ID,
    // not authorization to mint a new room. Persist the server-pinned manifest.
    const saved=await api('save',{...settingsPayload(),prevent_duplicate:false});
    if(saved.sha256!==result.sha256)throw Error('Imported model fingerprint changed; refusing to join');
    await refreshModels();
  }
  activatedRoom=selected.room_id;
  setText('evalModel',selected.model);notify(t('modelSelected'));
}
$('selectModel').addEventListener('click',safe('Select model',selectModel));$('modelTrain').addEventListener('click',safe('Select model',async()=>{await selectModel();navigate('training');}));
function sourcePayload(){return sources.map(s=>({...s,columns:{...s.columns}}));}
// New pretrained repository: do not inherit TinyStories' random genesis.
// This also enables the low-RAM streaming loader for newly selected HF models.
// Existing pinned rooms are set by applyManifest() and are never rewritten.
$('model').addEventListener('change',()=>{
  if(get('creationMode')==='weights' && get('model')!=='roneneldan/TinyStories-1M'
      && get('modelInitialization')==='random'){
    $('modelInitialization').value='pretrained';
    $('trainStreamingBase').value='on';
  }
});
function settingsPayload(){return {room:get('room'),model:get('model'),revision:get('revision'),task:get('task'),ws:get('ws'),seed:Number(get('seed')),initialization:get('modelInitialization'),creation_mode:get('creationMode'),sources:sourcePayload(),code_path:codePath,code_url:get('codeUrl'),tokenizer_model:get('tokenizerModel'),tokenizer_revision:get('tokenizerRevision'),adapter_kind:get('kind'),adapter_scope:get('adapterScope')};}
async function saveModel(){if(!sources.length)throw Error(t('noDataset'));const code=get('creationMode')==='code'?$('codeEditor').value:'';if(code.trim()){
    const bytes=new TextEncoder().encode(code);if(bytes.length>256*1024)throw Error('Code is over 256KiB');let bin='';for(let i=0;i<bytes.length;i+=4096)bin+=String.fromCharCode(...bytes.slice(i,i+4096));const result=await api('code',{name:'architecture.py',contents:btoa(bin)});codePath=result.path;setText('codeStatus','SHA-256 '+result.sha256);
  }const data=settingsPayload();data.prevent_duplicate=true;data.allow_duplicate=$('allowDuplicateModel').checked;const saved=await api('save',data);setText('fingerprint','SHA-256 '+saved.sha256);notify(t('modelSaved'));selected={room_id:data.room,model:data.model,revision:data.revision,datasets:sources.map(s=>s.repo),task:data.task,local:true,online:0};activatedRoom=data.room;$('allowDuplicateModel').checked=false;await refreshModels();void refreshAutoPlan().catch(()=>{});return saved;}
$('save').addEventListener('click',safe('Create model',async()=>{await saveModel();navigate('models');}));
$('newRoom').addEventListener('click',safe('New ID',async()=>{$('room').value=(await api('newRoom')).room;mixedPinnedPolicy=null;applyMixed({enabled:false});activatedRoom=null;selectedByUser=false;codePath='';setText('fingerprint','—');}));
function syncCreationMode(setDefault=true){const mode=get('creationMode');const init=$('modelInitialization');if(mode!=='weights'){init.value='config';init.disabled=true;}else{init.disabled=false;if(setDefault||init.value==='config')init.value='pretrained';}const tokFields=$('tokenizerFields');tokFields.hidden=mode!=='code';const editor=$('codeEditor');editor.disabled=false;editor.readOnly=false;$('codeUpload').disabled=false;if(mode==='code'&&!editor.value.trim()&&setDefault){editor.value='from transformers import GPT2Config\n\ndef build_config():\n    return GPT2Config(vocab_size=50257, n_positions=512, n_embd=256, n_layer=6, n_head=8)\n';}setText('codeStatus',mode==='code'?('Python Config · '+(codePath?'registered':'edit to define')):(mode==='architecture'?'Built-in HF architecture · no source execution':'Pretrained checkpoint'));}
$('creationMode').addEventListener('change',()=>{codePath='';syncCreationMode();});
// A click into the Python editor is an explicit opt-in to code-design mode.
$('codeEditor').addEventListener('focus',()=>{if(get('creationMode')!=='code'){$('creationMode').value='code';codePath='';syncCreationMode(false);}});
$('codeEditor').addEventListener('input',()=>{codePath='';setText('codeStatus','Unsaved Python changes');});
$('codeEditor').addEventListener('keydown',e=>{if(e.key!=='Tab'||e.ctrlKey||e.metaKey||e.altKey)return;e.preventDefault();const ed=e.currentTarget,start=ed.selectionStart,end=ed.selectionEnd;ed.setRangeText('    ',start,end,'end');ed.dispatchEvent(new Event('input',{bubbles:true}));});
$('registerRoom').addEventListener('click',safe('Publish model',async()=>{await saveModel();const v=await api('register',{room:get('room'),ws:get('ws'),room_public:get('visibility')==='public'});notify(v.public?'Public model registered':'Private model registered');}));
$('pickDatasets').addEventListener('click',()=>navigate('datasets'));
$('codeUpload').addEventListener('change',safe('Code',async()=>{const file=$('codeUpload').files[0];if(!file)return;if(file.size>256*1024)throw Error('Code is over 256KiB');$('codeEditor').value=await file.text();codePath='';}));
function appendDataset(entry){
  const opts={repo:String(entry.repo||'').trim(),revision:String(entry.revision||'main').trim(),config:String(entry.config||'').trim()||null,split:String(entry.split||'train').trim(),columns:entry.columns||{text:'text'},weight:Number(entry.weight)};
  if(!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$/.test(opts.repo))throw Error('Dataset must use namespace/repository on Hugging Face');
  if(!Number.isFinite(opts.weight)||opts.weight<=0||opts.weight>100)throw Error('Mix weight must be 0..100');
  if(sources.some(s=>s.repo===opts.repo&&s.config===opts.config&&s.split===opts.split&&s.revision===opts.revision)){notify(t('datasetDuplicate'));return false;}
  if(sources.length>=12)throw Error(t('datasetLimit'));
  sources.push(opts);drawSources();notify(t('datasetAdded')+' · '+opts.repo);return true;
}
function drawSources(){if($('mixPtDataset'))syncMixed();setText('draftCount',sources.length+' / 12');const root=$('draftDatasetCards');root.replaceChildren();if(!sources.length){root.append(create('div','empty',t('noDataset')));}sources.forEach((src,i)=>{const r=create('div','draft-row');const names=create('div','truncate');names.append(create('strong','',src.repo));names.append(create('div','tiny muted',src.split+' · ×'+src.weight));const del=create('button','', '×');del.setAttribute('aria-label','Remove dataset');del.addEventListener('click',()=>{sources.splice(i,1);drawSources();});r.append(names,del);root.append(r);});taglist($('newDatasetTags'),sources.map(s=>s.repo));}
$('addDataset').addEventListener('click',safe('Dataset',async()=>{const cols={};for(const v of get('columns').split(',')){const x=v.split('=');if(x.length!==2)throw Error('Expected text=text');cols[x[0].trim()]=x[1].trim();}appendDataset({repo:get('repo'),config:get('config')||null,revision:get('ds_revision'),split:get('split'),weight:Number(get('weight')),columns:cols});}));
function drawHub(){const pages=Math.max(1,Math.ceil(hubResults.length/5));datasetsPage=Math.min(datasetsPage,pages-1);const root=$('hfResults');root.replaceChildren();for(const x of hubResults.slice(datasetsPage*5,(datasetsPage+1)*5)){
  const item=create('div','hf-card');const meta=create('div','truncate');meta.append(create('strong','',x.id));meta.append(create('small','',Math.round(m(x.downloads)).toLocaleString()+' downloads'+(x.gated?' · gated':'')));
  const b=create('button','secondary',t('download'));b.type='button';b.addEventListener('click',safe('Dataset',async()=>{
    // Resolve actual Hub configs/splits instead of assuming that every repo has a train split.
    $('repo').value=x.id;$('ds_revision').value=x.revision||'main';
    const info=await api('inspect',{repo:x.id,revision:x.revision||'main'}).catch(e=>({error:e.message,variants:[]}));
    const variants=Array.isArray(info.variants)?info.variants:[];
    const cfgs=[...new Set(variants.map(v=>v.config))].filter(Boolean);
    const splits=[...new Set(variants.map(v=>v.split))].filter(Boolean);
    $('datasetConfigs').replaceChildren(...cfgs.map(c=>{const z=create('option');z.value=c;return z;}));
    $('datasetSplits').replaceChildren(...splits.map(c=>{const z=create('option');z.value=c;return z;}));
    $('config').value=cfgs.length===1?cfgs[0]:'';
    $('split').value=splits.includes('train')?'train':(splits[0]||'train');
    $('datasetInspectNote').textContent=info.error?('Hub inspection unavailable: '+info.error+'; check config/split before adding.'):(cfgs.length>1?'Multiple configurations available — choose a config and click Add dataset.':!variants.length?'No config/split metadata available — verify fields and click Add dataset.':'Hub config/split detected. Verify column mapping for your task.');
    if(cfgs.length>1||!variants.length){$('config').focus();return;}
    appendDataset({repo:x.id,revision:x.revision||'main',config:cfgs[0]||null,split:$('split').value,columns:{text:'text'},weight:1});
  }));item.append(meta,b);root.append(item);
 }if(!hubResults.length)root.append(create('div','empty',t('searchToStart')));setText('datasetCount',hubResults.length+' datasets');setPageControls('dataset',datasetsPage,pages,hubResults.length);}
let datasetsSearchSequence=0;
async function searchDatasets(){const ticket=++datasetsSearchSequence;setText('datasetCount','Searching...');const result=await api('hubDiscover',{role:'pt',query:get('datasetQuery'),scope:'recommended'});if(ticket!==datasetsSearchSequence)return;hubResults=(result.candidates||[]).map(x=>({...x,revision:x.revision||'main'}));datasetsPage=0;drawHub();}
$('searchDatasets').addEventListener('click',safe('Datasets',searchDatasets));$('datasetQuery').addEventListener('keydown',e=>{if(e.key==='Enter')void safe('Datasets',searchDatasets)();});$('datasetPrev').addEventListener('click',()=>{datasetsPage=Math.max(0,datasetsPage-1);drawHub();});$('datasetNext').addEventListener('click',()=>{datasetsPage++;drawHub();});
// v70: one accessible browser for purpose-specific Hub suggestions.
// Default candidates are fetched immediately on open; they are NEVER treated as
// compatibility-verified until model loading / dataset mapping succeeds.
const hubPurposes={pt:'PRETRAINING · テキストコーパス',sft:'SFT · 指示応答データ',ppo:'PPO · プロンプト / 選好データ',reward:'REWARD · 回答評価モデル',critic:'CRITIC · 価値関数モデル'};
let hubPickerRole=null,hubPickerSequence=0,hubPickerReturn=null,hubPickerChoosing=false;
function closeHubPicker(){if($('hubPicker').hidden)return;$('hubPicker').hidden=true;hubPickerRole=null;hubPickerSequence++;hubPickerChoosing=false;clearTimeout(hubPickerDebounce);if(hubPickerReturn&&typeof hubPickerReturn.focus==='function')hubPickerReturn.focus();}
function hubPickerStatus(msg){setText('hubPickerNotice',msg);}
async function loadHubPicker(){
 const role=hubPickerRole;if(!role)return;
 const ticket=++hubPickerSequence;const query=get('hubPickerQuery'),scope=get('hubPickerScope');
 setText('hubPickerCount','Loading…');$('hubPickerResults').replaceChildren(create('div','empty','Hubの候補を読み込み中…'));
 hubPickerStatus('用途に対応する公開メタデータから候補を抽出しています。実行互換性は未確認です。');
 try{
  const result=await api('hubDiscover',{role,query,scope});if(ticket!==hubPickerSequence||role!==hubPickerRole)return;
  const results=result.candidates||[];const root=$('hubPickerResults');root.replaceChildren();
  if(!results.length)root.append(create('div','empty','候補がありません。検索語を変更するか「Hubを広く検索」を選んでください。'));
  for(const item of results){
   const row=create('button','hub-candidate');row.type='button';
   const left=create('div','hub-candidate-content');left.append(create('strong','',item.id),create('small','',item.reason||'形式の確認が必要です'));
   const aside=create('div','hub-candidate-aside');aside.append(create('span','',item.kind==='dataset'?'DATASET':'MODEL'));
   aside.append(create('span','',Number(item.downloads||0).toLocaleString()+' downloads'));
   if(item.gated)aside.append(create('span','hub-gated','要アクセス許可'));
   row.append(left,aside);row.addEventListener('click',safe('Hub candidate',async()=>{
    if(hubPickerChoosing)return;
    const selectedAt=hubPickerSequence;
    hubPickerChoosing=true;row.disabled=true;
    try{await chooseHubCandidate(role,item,selectedAt);}
    finally{row.disabled=false;if(hubPickerSequence===selectedAt)hubPickerChoosing=false;}
   }));root.append(row);
  }
  setText('hubPickerCount',results.length+' candidates · '+(scope==='recommended'?'用途別に絞り込み':'広範囲'));
  hubPickerStatus('検索結果は用途から推定した候補です。SFT/PPOは列設定、Reward/Criticは学習済みのスカラーヘッドを別途検証してください。'+(result.partial_errors?.length?' · 部分的なHub検索失敗あり':'') );
 }catch(e){if(ticket!==hubPickerSequence)return;setText('hubPickerCount','検索失敗');$('hubPickerResults').replaceChildren(create('div','empty','Hubに接続できません。ネットワークまたはログインを確認してください。'));hubPickerStatus(String(e.message||e));}
}
async function openHubPicker(role,trigger){if(!hubPurposes[role])throw Error('Invalid Hub purpose');hubPickerReturn=trigger||null;hubPickerRole=role;$('hubPicker').hidden=false;
 setText('hubPickerPurpose','HUGGING FACE / '+role.toUpperCase());setText('hubPickerTitle',hubPurposes[role]);
 setText('hubPickerHelp',role==='critic'?'学習済みの価値ヘッドが必要です。Reward ModelとCriticは同一ではありません。':role==='ppo'?'PPOではプロンプト列が必要です。外部Reward Modelを使う場合、正解列は不要です。':role==='sft'?'プロンプトと回答の列を確認してください。':role==='reward'?'スカラー報酬ヘッドの存在を確認してください。':'テキスト列と学習splitを確認してください。');
 $('hubPickerQuery').value='';$('hubPickerScope').value='recommended';$('hubPickerQuery').focus();await loadHubPicker();
}
async function chooseHubCandidate(role,item,selectedAt){
 const stillCurrent=()=>!$('hubPicker').hidden&&hubPickerRole===role&&hubPickerSequence===selectedAt;
 if(role==='pt'){
  const info=await api('mixPinDataset',{repo:item.id});if(!stillCurrent())return;const train=(info.variants||[]).filter(v=>v.split==='train');
  const options=train.length?train:(info.variants||[]);$('repo').value=info.repo;$('ds_revision').value=info.revision;
  if(options.length===1){$('config').value=options[0].config;$('split').value=options[0].split;}
  else if(options.length){$('config').value='';$('split').value='train';}
  $('datasetInspectNote').textContent='モデルに追加する前に、Config・Split・text列を確認してください。'+(options.length>1?' 複数のconfigがあります。':'');
  closeHubPicker();$('repo').focus();return;
 }
 if(role==='sft'||role==='ppo'){
  const info=await api('mixPinDataset',{repo:item.id});if(!stillCurrent())return;const tag=role==='sft'?'Sft':'Ppo';
  $('mix'+tag+'Source').value='hf';$('mix'+tag+'HfRepo').value=info.repo;$('mix'+tag+'HfRevision').value=info.revision;
  const variants=info.variants||[],train=variants.filter(v=>v.split==='train');const chosen=train.length===1?train[0]:variants.length===1?variants[0]:null;
  if(chosen){$('mix'+tag+'HfConfig').value=chosen.config;$('mix'+tag+'HfSplit').value=chosen.split;}
  else{$('mix'+tag+'HfConfig').value='';$('mix'+tag+'HfSplit').value='train';}
  $('mix'+tag+'Advanced').open=!chosen;setText('mix'+tag+'HfStatus','Commit固定: '+info.revision.slice(0,12)+' · '+(chosen?'config/splitを設定しました。列名を確認してください。':'複数のconfigがあります。config/splitを確認してください。'));
  syncMixed();closeHubPicker();$('mix'+tag+'HfPrompt').focus();return;
 }
 const info=await api('rewardInspect',{repo:item.id});if(!stillCurrent())return;if(role==='reward'){
  $('mixPpoReward').value='hf_model';$('mixRewardModelId').value=info.id;$('mixRewardRevision').value=info.revision;
  syncMixed();closeHubPicker();$('mixRewardOutput').focus();
 }else{
  $('mixCriticKind').value='hf';$('mixCriticHfRepo').value=info.id;$('mixCriticHfRevision').value=info.revision;
  syncMixed();closeHubPicker();$('mixCriticHfLoader').focus();
 }
 notify('選択しました · '+info.id+' @ '+info.revision.slice(0,12)+' · 実行互換性は要検証');
}
$('hubPickerClose').addEventListener('click',closeHubPicker);$('hubPickerBackdrop').addEventListener('click',closeHubPicker);
$('hubPickerSearch').addEventListener('click',safe('Hub browser',loadHubPicker));$('hubPickerScope').addEventListener('change',safe('Hub filter',loadHubPicker));
let hubPickerDebounce=null;
$('hubPickerQuery').addEventListener('input',()=>{hubPickerSequence++;clearTimeout(hubPickerDebounce);hubPickerDebounce=setTimeout(()=>{void safe('Hub browser',loadHubPicker)();},480);});
$('hubPickerQuery').addEventListener('keydown',e=>{if(e.key==='Enter'){clearTimeout(hubPickerDebounce);void safe('Hub browser',loadHubPicker)();}});
if(typeof document.addEventListener==='function')document.addEventListener('keydown',e=>{
 if($('hubPicker').hidden)return;
 if(e.key==='Escape'){e.preventDefault();closeHubPicker();return;}
 if(e.key==='Tab'){
  const items=Array.from($('hubPicker').querySelectorAll('input,select,button')).filter(x=>!x.disabled&&x.offsetParent!==null);
  if(!items.length)return;
  if(e.shiftKey&&document.activeElement===items[0]){e.preventDefault();items[items.length-1].focus();}
  else if(!e.shiftKey&&document.activeElement===items[items.length-1]){e.preventDefault();items[0].focus();}
 }
});
$('ptHubBrowse').addEventListener('click',safe('PT Hugging Face',()=>openHubPicker('pt',$('ptHubBrowse'))));

// Local paths are per-peer and must never appear in a public room manifest.
let mixedPinnedPolicy=null;
const mixedIds={sft_hf_repo:'mixSftHfRepo',sft_hf_revision:'mixSftHfRevision',sft_hf_config:'mixSftHfConfig',sft_hf_split:'mixSftHfSplit',sft_hf_prompt_column:'mixSftHfPrompt',sft_hf_response_column:'mixSftHfOther',ppo_hf_repo:'mixPpoHfRepo',ppo_hf_revision:'mixPpoHfRevision',ppo_hf_config:'mixPpoHfConfig',ppo_hf_split:'mixPpoHfSplit',ppo_hf_prompt_column:'mixPpoHfPrompt',ppo_hf_reference_column:'mixPpoHfOther',sft_jsonl:'mixSftPath',ppo_jsonl:'mixPpoPath',critic_init:'mixCriticInit',critic_hf_repo:'mixCriticHfRepo',critic_hf_revision:'mixCriticHfRevision',critic_hf_loader:'mixCriticHfLoader',critic_hf_device:'mixCriticHfDevice',critic_hf_max_length:'mixCriticHfMaxLength',
  pt_weight:'mixPtWeight',sft_weight:'mixSftWeight',ppo_weight:'mixPpoWeight',
  ppo_clip:'mixPpoClip',ppo_kl_coef:'mixPpoKl',ppo_value_coef:'mixValueCoef',
  ppo_gamma:'mixGamma',ppo_gae_lambda:'mixGaeLambda',ppo_max_tokens:'mixPpoTokens',
  ppo_reuse:'mixPpoReuse',ppo_reward:'mixPpoReward',critic_offpolicy:'mixCriticOffpolicy',
  reward_model_id:'mixRewardModelId',reward_model_revision:'mixRewardRevision',
  reward_model_device:'mixRewardDevice',reward_model_loader:'mixRewardLoader',reward_model_output:'mixRewardOutput',reward_model_format:'mixRewardFormat',
  reward_model_index:'mixRewardIndex',reward_model_scale:'mixRewardScale',
  reward_model_max_length:'mixRewardMaxLength',critic_kind:'mixCriticKind',
  critic_hidden:'mixCriticHidden',critic_lr:'mixCriticLr'};
const mixedNumeric=new Set(['pt_weight','sft_weight','ppo_weight','ppo_clip','ppo_kl_coef',
  'ppo_value_coef','ppo_gamma','ppo_gae_lambda','ppo_max_tokens','ppo_reuse',
  'critic_hidden','critic_lr','critic_hf_max_length','reward_model_index','reward_model_scale','reward_model_max_length']);
const mixedInt=new Set(['ppo_max_tokens','ppo_reuse','critic_hidden','critic_hf_max_length','reward_model_index','reward_model_max_length']);
function syncMixed(){for(const x of ['Sft','Ppo']){const hf=get('mix'+x+'Source')==='hf';$('mix'+x+'Hub').hidden=!hf;$('mix'+x+'Local').hidden=hf;}const on=get('mixEnabled')==='on';$('mixFields').hidden=!on;const k=get('mixCriticKind');$('mixCriticHidden').disabled=k!=='mlp';$('mixCriticHub').hidden=k!=='hf';$('mixCriticLocalInit').hidden=k==='hf';
  $('mixRewardFields').hidden=get('mixPpoReward')!=='hf_model';setText('mixPtDataset',sources.map(x=>x.repo+' ['+x.split+']').join(', ')||'No PT dataset');
 const summary=[['PT',sources.length?sources.length+' dataset(s)':'not selected'],['SFT',get('mixSftSource')==='hf'?get('mixSftHfRepo')||'not selected':'local JSONL'],['PPO',get('mixPpoSource')==='hf'?get('mixPpoHfRepo')||'not selected':'local JSONL'],['Reward',get('mixPpoReward')==='hf_model'?get('mixRewardModelId')||'not selected':get('mixPpoReward')],['Critic',k==='hf'?get('mixCriticHfRepo')||'not selected':k.toUpperCase()]];
 setText('mixSetupSummary',summary.map(x=>x[0]+' : '+x[1]).join('  ·  '));}
function mixedPayload(){
 const enabled=get('mixEnabled')==='on';if(!enabled)return {enabled:false};
 const data={enabled:true};
 for(const [field,id] of Object.entries(mixedIds)){
   const raw=get(id);data[field]=mixedNumeric.has(field)?Number(raw):raw;
   if(mixedNumeric.has(field)&&(!raw||!Number.isFinite(data[field])||(mixedInt.has(field)&&!Number.isInteger(data[field]))))throw Error('Invalid '+field);
 }
 data.distributed_critic=get('mixDistributedCritic')!=='off';
 data.critic_rollout_gossip=get('mixDistributedCritic')==='rollout';
 data.reward_model_idle_offload=get('mixRewardIdleOffload')==='on';
 for(const [role,tag] of [['sft','Sft'],['ppo','Ppo']]){if(get('mix'+tag+'Source')==='hf')data[role+'_jsonl']='';else{for(const key of Object.keys(data))if(key.startsWith(role+'_hf_'))data[key]='';}}
 if(!data.sft_jsonl&&!data.ppo_jsonl&&!data.sft_hf_repo&&!data.ppo_hf_repo&&data.pt_weight<=0)throw Error('Select an SFT/PPO dataset or assign a positive PT weight');
 if(!mixedPinnedPolicy){
   if(!data.sft_jsonl&&!data.sft_hf_repo)data.sft_weight=0;
   if(!data.ppo_jsonl&&!data.ppo_hf_repo)data.ppo_weight=0;
 }
 if(get('task')!=='causal-lm')throw Error('Mixed PT/SFT/PPO currently requires causal-lm');
 return data;
}
function applyMixed(data){$('mixEnabled').value=data?.enabled?'on':'off';for(const [field,id] of Object.entries(mixedIds)){
  const value=data?.[field];const defaults={pt_weight:1,sft_weight:1,ppo_weight:.2,ppo_clip:.2,
    ppo_kl_coef:.02,ppo_value_coef:.5,ppo_gamma:.99,ppo_gae_lambda:.95,
    ppo_max_tokens:8,ppo_reuse:2,ppo_reward:'token_match',critic_offpolicy:'auto',
    reward_model_id:'',reward_model_revision:'',reward_model_device:'cpu',reward_model_loader:'sequence_classification',
    reward_model_output:'logits',reward_model_format:'plain',reward_model_index:0,reward_model_scale:1,reward_model_max_length:512,critic_kind:'mlp',critic_hidden:32,critic_lr:.001,critic_hf_repo:'',critic_hf_revision:'',critic_hf_loader:'sequence_classification',critic_hf_device:'cpu',critic_hf_max_length:512};
  const hfDefaults={sft_hf_split:'train',sft_hf_prompt_column:'prompt',sft_hf_response_column:'response',ppo_hf_split:'train',ppo_hf_prompt_column:'prompt',ppo_hf_reference_column:'reference'};
  $(id).value=value??defaults[field]??hfDefaults[field]??'';
 }$('mixDistributedCritic').value=data?.critic_rollout_gossip?'rollout':data?.distributed_critic?'on':'off';$('mixRewardIdleOffload').value=data?.reward_model_idle_offload?'on':'off';for(const [role,tag] of [['sft','Sft'],['ppo','Ppo']]){$('mix'+tag+'Source').value=data?.[role+'_hf_repo']?'hf':data?.[role+'_jsonl']?'local':'hf';}syncMixed();}
async function reloadMixed(room){
 const data=await api('mixedSettings',{room});mixedPinnedPolicy=data.pinned_policy||null;
 if(mixedPinnedPolicy&&!data.enabled){
   data.enabled=true;
   const p=mixedPinnedPolicy;
   if(Array.isArray(p.weights)&&p.weights.length===3){
     [data.pt_weight,data.sft_weight,data.ppo_weight]=p.weights;
   }
   for(const key of ['ppo_clip','ppo_kl_coef','ppo_value_coef','ppo_gamma',
                     'ppo_gae_lambda','ppo_max_tokens','ppo_reuse','ppo_reward']){
     if(p[key]!==undefined)data[key]=p[key];
   }
   if(p.reward_model){data.reward_model_id=p.reward_model.repo;data.reward_model_revision=p.reward_model.revision;
     data.reward_model_output=p.reward_model.output;data.reward_model_loader=p.reward_model.loader;data.reward_model_format=p.reward_model.format;data.reward_model_index=p.reward_model.index;
     data.reward_model_scale=p.reward_model.scale;data.reward_model_max_length=p.reward_model.max_length;
     data.reward_model_device='cpu';}
   data.distributed_critic=['signed-root-critic-gossip-v78','checkpoint-verified-trajectory-is-v81'].includes(p.critic_distribution);
   data.critic_rollout_gossip=p.critic_distribution==='checkpoint-verified-trajectory-is-v81';
   data.critic_offpolicy=p.critic_offpolicy==='unclipped-trajectory-is-mc-v79'?'trajectory_is':'gae';
   if(p.critic){data.critic_kind=p.critic.kind;data.critic_hidden=p.critic.hidden;data.critic_lr=p.critic.lr; if(p.critic.hf){data.critic_hf_repo=p.critic.hf.repo;data.critic_hf_revision=p.critic.hf.revision;data.critic_hf_loader=p.critic.hf.loader;data.critic_hf_max_length=p.critic.hf.max_length;data.critic_hf_device='cpu';}}
 }
 applyMixed(data);
}
for(const [role,tag] of [['sft','Sft'],['ppo','Ppo']]){
 $('mix'+tag+'Source').addEventListener('change',syncMixed);
 $('mix'+tag+'HfSearch').addEventListener('click',safe(role.toUpperCase()+' Hub',()=>openHubPicker(role,$('mix'+tag+'HfSearch'))));
 $('mix'+tag+'HfResults').addEventListener('change',()=>{
   const select=$('mix'+tag+'HfResults'),opt=select.selectedOptions[0];if(!select.value)return;
   $('mix'+tag+'HfRepo').value=select.value;
   if(/^[a-fA-F0-9]{40}$/.test(opt.dataset.sha||''))$('mix'+tag+'HfRevision').value=opt.dataset.sha;
 });
 $('mix'+tag+'HfInspect').addEventListener('click',safe('Inspect '+role.toUpperCase()+' dataset',async()=>{
   const repo=get('mix'+tag+'HfRepo');const revision=get('mix'+tag+'HfRevision');
   const info=await api('mixPinDataset',{repo});
   $('mix'+tag+'HfRevision').value=info.revision;const variants=info.variants||[];
   if(variants.length===1){$('mix'+tag+'HfConfig').value=variants[0].config;$('mix'+tag+'HfSplit').value=variants[0].split;}
   setText('mix'+tag+'HfStatus',variants.length?('Available: '+variants.slice(0,12).map(x=>x.config+'/'+x.split).join(', ')):'Config/split unavailable from metadata. Check the dataset card and fill in manually.');
 }));
}
$('mixEnabled').addEventListener('change',syncMixed);
$('mixCriticKind').addEventListener('change',syncMixed);
$('mixDistributedCritic').addEventListener('change',syncMixed);
$('mixCriticSearch').addEventListener('click',safe('HF Critic Hub',()=>openHubPicker('critic',$('mixCriticSearch'))));
$('mixCriticSearchResults').addEventListener('change',()=>{
 const sel=$('mixCriticSearchResults');if(!sel.value)return;
 $('mixCriticHfRepo').value=sel.value;const sha=sel.selectedOptions[0]?.dataset.sha||'';
 if(/^[a-fA-F0-9]{40}$/.test(sha))$('mixCriticHfRevision').value=sha;
});
$('mixCriticPin').addEventListener('click',safe('Value Critic revision',async()=>{
 const found=await api('rewardInspect',{repo:get('mixCriticHfRepo')});
 $('mixCriticHfRevision').value=found.revision;
 notify('Pinned Critic '+found.id+' @ '+found.revision.slice(0,12));
}));

$('mixPpoReward').addEventListener('change',syncMixed);
$('mixRewardSearch').addEventListener('click',safe('HF Reward Hub',()=>openHubPicker('reward',$('mixRewardSearch'))));
$('mixRewardSearchResults').addEventListener('change',()=>{
  const select=$('mixRewardSearchResults');const opt=select.selectedOptions[0];
  if(!select.value)return;
  $('mixRewardModelId').value=select.value;
  if(opt.dataset.sha)$('mixRewardRevision').value=opt.dataset.sha;
});
$('mixRewardPin').addEventListener('click',safe('Reward Model revision',async()=>{
  const found=await api('rewardInspect',{repo:get('mixRewardModelId')});
  $('mixRewardRevision').value=found.revision;
  notify('Pinned '+found.id+' @ '+found.revision.slice(0,12));
}));

for(const [button,target,kind] of [['pickMixSft','mixSftPath','sft'],['pickMixPpo','mixPpoPath','ppo'],['pickMixCritic','mixCriticInit','critic']]){
 $(button).addEventListener('click',safe('Browse',async()=>{const r=await api('pickMixFile',{kind});if(r.path)$(target).value=r.path;}));
}
function trainingOptions(plan){if(!plan||!plan.can_train)throw Error('This device cannot train the pinned shared rank; relay-only participation is supported when a suitable peer worker is online.');if(!Number.isInteger(plan.max_length)||!Number.isInteger(plan.local_steps))throw Error('Automatic training policy not available');return {room:get('room'),base:get('base'),ws:get('ws'),kind:plan.kind,rank:plan.rank,reinit_every:Number(get('trainReinitEvery')),room_public:get('visibility')==='public',optimizer:get('trainOptimizer'),initialization:(get('creationMode')==='weights'?get('modelInitialization'):'config'),lr:Number(get('trainLr')),local_steps:plan.local_steps,max_length:plan.max_length,streaming_base:get('trainStreamingBase')==='on',disk_cache_fraction:Number(get('trainDiskCacheFraction')),wide_lora_rank:plan.wide_lora_rank,rounds:0,push_fanout:Number(get('trainPushFanout')),journal_sync:get('trainJournalSync'),max_journal_gb:0,mixed_training:mixedPayload()};}
async function refreshAutoPlan(requireFresh=false){
  const sequence=++autoPlanRequest;
  const payload={room:get('room'),model:get('model'),revision:get('revision'),kind:get('kind')};
  const plan=await api('plan',payload);
  if(sequence!==autoPlanRequest&&!requireFresh)return autoPlan||plan;
  autoPlan=plan;
  setText('trainAutoRank',plan.rank+' · '+(plan.rank_pinned?'shared':'estimated')+(plan.can_train?'':' · relay only'));
  setText('trainAutoContext',plan.max_length+' tokens');
  if(!run.running)setText('trainAutoPublish',plan.local_steps+' base');
  if(!run.running)setText('trainAutoPeers','Automatic (live)');
  setText('trainAutoNote',plan.reason||t('autoNote'));
  return plan;
}
$('kind').addEventListener('change',()=>{void safe('Auto tuning',refreshAutoPlan)();});
async function startOrResume(){
  if(run.running){if(run.relay_only){await api('resume');notify(t('trainResumed'));}else notify(t('alreadyTraining'));await poll();return;}
  const payload=settingsPayload();
  if(!activatedRoom && selectedByUser && selected?.local){await selectModel();Object.assign(payload,settingsPayload());}
  if(!activatedRoom||activatedRoom!==payload.room)throw Error(lang==='ja'?'MODELSで既存モデルを選択するか、NEW MODELで明示的に作成してください。学習開始だけでは新規作成しません。':'Select a saved model in MODELS, or explicitly create one in NEW MODEL. Starting training never creates a model.');
  if(selected&&selected.room_id===payload.room&&['unavailable','unsupported'].includes(selected.compatibility?.status))throw Error(t('cannotTrain')+': '+selected.compatibility.reason);
  const listing=await api('localModels');
  if(!(listing.models||[]).some(x=>x.room_id===payload.room))throw Error('Model is not saved locally. Select or create the model first.');
  const old=(await api('select',{room:payload.room,ws:payload.ws})).manifest;
  const same=old.model.id===payload.model&&old.model.revision===payload.revision&&old.model.architecture===payload.task&&old.datasets.seed===payload.seed&&JSON.stringify(old.datasets.sources)===JSON.stringify(payload.sources);
  if(!same)throw Error('Model is immutable. Re-select it in MODELS before starting.');
  const plan=await refreshAutoPlan(true);const out=await api('start',trainingOptions(plan));notify(t('trainStarted')+' · PID '+out.pid);await poll();
}
async function pause(){const r=await api('pause');if(r.running)notify(t('trainPaused'));else notify(t('idle'));await poll();}
async function disconnect(){await api('disconnect');notify(t('disconnecting'));await poll();}
$('start').addEventListener('click',safe('Training',startOrResume));$('stop').addEventListener('click',safe('Pause',pause));$('disconnect').addEventListener('click',safe('Disconnect',disconnect));$('dockToggle').addEventListener('click',safe('Training',async()=>run.running&&!run.relay_only?pause():startOrResume()));$('dockTraining').addEventListener('click',()=>navigate('training'));
$('trainOptimizer').addEventListener('change',()=>{if(get('trainOptimizer')==='lora-muon')$('kind').value='lora';});
for(const [id,kind] of [['openCheckpoints','checkpoints'],['openModels','models'],['openGenesis','genesis']])$(id).addEventListener('click',safe('Files',async()=>{if(!get('room'))throw Error(t('needSelect'));const r=await api('open',{room:get('room'),kind});setText('storageStatus',r.path);notify(r.path);}));
$('copyLogs').addEventListener('click',safe('Logs',async()=>{const r=await api('copy');notify(r.chars+' characters copied');}));
$('saveLogs').addEventListener('click',safe('Logs',async()=>{const r=await api('saveLog');notify(r.path);}));
$('captureSnapshot').addEventListener('click',safe('Snapshot',async()=>{const r=await api('captureSnapshot',{room:get('room')});setText('evalCheckpoint','SAVE REQUEST QUEUED…');notify(r.message||'Snapshot requested');await updateCheckpoints();}));
let queuedEval=null;
$('runEval').addEventListener('click',safe('EVAL',async()=>{
 if(!get('room'))throw Error(t('needSelect'));
 const options={room:get('room'),scope:get('evalScope'),prompt:$('evalPrompt').value,precision:get('evalPrecision'),max_tokens:Number(get('evalMaxTokens'))};
 let r;try{r=await api('eval',options);}catch(e){setText('evalStatus','ERROR');setText('evalResult','EVAL: '+String(e?.message||e));throw e;}
 if(r.pending_snapshot){queuedEval=options;setText('evalStatus','SAVING');setText('evalResult','Preparing a verified learned checkpoint before inference…');setText('evalMeta','Kagami trained weights');return;}
 queuedEval=null;setText('evalStatus','RUNNING');setText('evalResult',t('evalRunning'));setText('evalMeta',r.scope);await pollEval();}));
$('cancelEval').addEventListener('click',safe('Cancel',async()=>{queuedEval=null;await api('cancelEval');setText('evalStatus','CANCELLED');await pollEval();}));
async function pollEval(){
 if(queuedEval){
   const status=await api('snapshotStatus',{room:queuedEval.room});
   if(status.state==='error'){
     setText('evalStatus','SAVE ERROR');setText('evalResult',status.error||'Snapshot export failed. See TRAINING logs.');queuedEval=null;return;
   }
   if(status.state==='complete'){
     const payload=queuedEval;queuedEval=null;
     const next=await api('eval',payload);
     if(next.pending_snapshot){queuedEval=payload;return;}
     setText('evalStatus','RUNNING');setText('evalResult',t('evalRunning'));
   }else{setText('evalStatus','SAVING');return;}
 }
 const state=await api('evalStatus');if(state.running){setText('evalStatus','RUNNING');return;}if(!state.done)return;
 setText('evalStatus',state.exit_code===0&&!state.result?.error?'DONE':'ERROR');
 setText('evalResult',state.result?.text??(state.result?.error ? (state.result.error + (state.result.hint ? '\n\n'+state.result.hint : '')) : null)??state.logs??'—');
 setText('evalMeta',state.result?.error?String(state.logs||'').slice(-1800):(state.result?.scope??'')+(state.result?.precision?' · '+state.result.precision.toUpperCase():''));
}
async function updateCheckpoints(){if(!get('room'))return;
 const room=get('room');const [info,progress]=await Promise.all([
   api('evalCheckpoints',{room}),api('snapshotStatus',{room}).catch(()=>({state:'idle'}))]);
 const cp=info.checkpoints?.[0];
 const checkpoint=cp?('SAVED CHECKPOINT #'+cp.seq+' · '+(cp.bytes/1073741824).toFixed(2)+' GiB · '+cp.tensors+' tensors'):'NO SAVED LEARNED WEIGHTS';
 const status=progress.state==='error'?'SAVE FAILED: '+(progress.error||'See TRAINING logs'):
              progress.state==='saving'?'SAVING WEIGHTS… '+checkpoint:
              progress.state==='queued'?'SAVE REQUEST QUEUED… '+checkpoint:checkpoint;
 setText('evalCheckpoint',status);
 if(progress.state==='error')setText('evalStatus','SAVE ERROR');
}
// Actual telemetry samples, not generated chart data or inferred validation metrics.
const NS='http://www.w3.org/2000/svg';
function svgEl(name,attrs={},txt){const n=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attrs))n.setAttribute(key,String(value));if(txt!==undefined)n.textContent=String(txt);return n;}
function chartBounds(values,field){
  let lo=Math.min(...values),hi=Math.max(...values);
  const isAccuracy=field==='accuracy';
  // Top-1 now has an ADAPTIVE axis, still bounded by the mathematical range.
  // Avoid a zero-height range when all samples are identical.
  const minSpan=isAccuracy?0.08:Math.max(0.05,Math.abs(lo)*0.04,Math.abs(hi)*0.04);
  const delta=Math.max(hi-lo,minSpan);
  const padding=delta*0.09;
  let min=lo-padding,max=hi+padding;
  if(isAccuracy){
    min=Math.max(0,min);max=Math.min(1,max);
    if(max-min<minSpan){const center=(lo+hi)*0.5;
      min=Math.max(0,center-minSpan*0.5);
      max=Math.min(1,Math.max(min+minSpan,center+minSpan*0.5));
      min=Math.max(0,Math.min(min,max-minSpan));}
  }else{
    min=Math.min(min,lo);max=Math.max(max,hi);
    if(min>=max){min=lo-minSpan/2;max=hi+minSpan/2;}
    // Cross-entropy loss is non-negative, but do not conceal genuinely
    // negative values for alternative objectives.
    if(lo>=0)min=Math.max(0,min);
  }
  return {min,max};
}
function chartAxisNumber(value,field){
  return field==='accuracy'?(value*100).toFixed(1)+'%':
      (Math.abs(value)>=100?value.toFixed(1):value.toFixed(Math.abs(value)<1?3:2));
}
function chartPoints(field){
 const available=chartTelemetry.length?chartTelemetry:latestTelemetry;
 const period=Number(get('chartWindow'));
 const fresh=latestTelemetry.at(-1);
 const high=Number(fresh?.elapsed_s);
 // Enforce the selected time range even while appending the live tail.
 const cutoff=period&&Number.isFinite(high)?high-period:-Infinity;
 const combined=available.concat(latestTelemetry.slice(-8));
 const unique=new Map();
 for(const s of combined){
  const x=Number(s?.steps),y=Number(s?.[field]),elapsed=Number(s?.elapsed_s);
  if(!Number.isFinite(x)||s?.[field]==null||!Number.isFinite(y))continue;
  if(period&&Number.isFinite(elapsed)&&elapsed<cutoff)continue;
  unique.set(x,{x,y});
 }
 return Array.from(unique.values()).sort((a,b)=>a.x-b.x);
}
function plot(id,field){
 const svg=$(id);svg.replaceChildren();
 const pts=chartPoints(field);
 if(!pts.length){svg.append(svgEl('text',{x:320,y:82,fill:'#91bfa5','text-anchor':'middle','font-size':13},'NO MEASUREMENTS YET'));return;}
 const x0=pts[0].x,rawX1=pts[pts.length-1].x,x1=rawX1===x0?x0+1:rawX1;
 const {min:y0,max:y1}=chartBounds(pts.map(p=>p.y),field);
 const top=19,bottom=132,left=74,right=622;
 const xx=v=>left+(v-x0)/(x1-x0)*(right-left);
 const yy=v=>top+(y1-v)/(y1-y0)*(bottom-top);
 // Horizontal guide lines and accurate upper/lower Y-axis labels.
 for(let i=0;i<=4;i++){
  const y=top+(bottom-top)*i/4,val=y1-(y1-y0)*i/4;
  svg.append(svgEl('line',{x1:left,y1:y,x2:right,y2:y,stroke:i===4?'#426951':'#234435','stroke-width':i===4?1.1:.8,'stroke-dasharray':i===4?'':'3 5'}));
  svg.append(svgEl('text',{x:left-8,y:y+4,'text-anchor':'end',fill:'#b2d6bc','font-size':12},chartAxisNumber(val,field)));
 }
 for(let i=0;i<=4;i++){
  const x=left+(right-left)*i/4,val=x0+(x1-x0)*i/4;
  if(i>0&&i<4)svg.append(svgEl('line',{x1:x,y1:top,x2:x,y2:bottom,stroke:'#1d3629','stroke-dasharray':'2 7'}));
  svg.append(svgEl('text',{x,y:155,'text-anchor':i===0?'start':i===4?'end':'middle',fill:'#a7c4b0','font-size':11},Math.round(val).toLocaleString() ));
 }
 const coordinates=pts.map(p=>xx(p.x)+','+yy(p.y));
 const last=pts[pts.length-1];
 const poly='M'+coordinates.join(' L');
 const defs=svgEl('defs');const gradient=svgEl('linearGradient',{id:id+'-fill',x1:'0%',y1:'0%',x2:'0%',y2:'100%'});
 gradient.append(svgEl('stop',{offset:'0%','stop-color':'#10b981','stop-opacity':'.24'}),svgEl('stop',{offset:'100%','stop-color':'#10b981','stop-opacity':'0'}));defs.append(gradient);svg.append(defs);
 svg.append(svgEl('path',{d:poly+' L'+xx(last.x)+','+bottom+' L'+xx(pts[0].x)+','+bottom+' Z',fill:'url(#'+id+'-fill)'}));
 svg.append(svgEl('path',{d:poly,fill:'none',stroke:'#35e0a0','stroke-width':2.7,'stroke-linecap':'round','stroke-linejoin':'round'}));
 svg.append(svgEl('circle',{cx:xx(last.x),cy:yy(last.y),r:4.2,fill:'#e1ffeb',stroke:'#10b981','stroke-width':2}));
 // Hover/touch inspection shows actual raw data, never interpolated values.
 const inspectLine=svgEl('line',{x1:left,y1:top,x2:left,y2:bottom,stroke:'#c7f8da','stroke-dasharray':'4 4',visibility:'hidden'});
 const inspectDot=svgEl('circle',{cx:left,cy:top,r:4.5,fill:'#e8fff0',stroke:'#075b3d','stroke-width':1.5,visibility:'hidden'});
 const inspectText=svgEl('text',{x:right-4,y:top+13,'text-anchor':'end',fill:'#e4ffed','font-size':12,'font-weight':'600',visibility:'hidden'});
 svg.append(inspectLine,inspectDot,inspectText);
 const overlay=svgEl('rect',{x:left,y:top,width:right-left,height:bottom-top,fill:'transparent','pointer-events':'all'});
 function inspect(e){
   const rect=svg.getBoundingClientRect?.();
   if(!rect||!rect.width)return;
   const fraction=Math.min(1,Math.max(0,((e.clientX-rect.left)/rect.width*650-left)/(right-left)));
   const target=x0+fraction*(x1-x0);
   let lo=0,hi=pts.length-1;
   while(lo<hi){const mid=(lo+hi)>>1;if(pts[mid].x<target)lo=mid+1;else hi=mid;}
   const selected=lo>0&&Math.abs(pts[lo-1].x-target)<Math.abs(pts[lo].x-target)?pts[lo-1]:pts[lo];
   const px=xx(selected.x),py=yy(selected.y);
   for(const element of [inspectLine,inspectDot,inspectText])element.setAttribute('visibility','visible');
   inspectLine.setAttribute('x1',px);inspectLine.setAttribute('x2',px);
   inspectDot.setAttribute('cx',px);inspectDot.setAttribute('cy',py);
   inspectText.textContent='STEP '+Math.round(selected.x).toLocaleString()+'  •  '+chartAxisNumber(selected.y,field);
 }
 overlay.addEventListener('pointermove',inspect);
 overlay.addEventListener('pointerdown',inspect);
 overlay.addEventListener('pointerleave',()=>{for(const el of [inspectLine,inspectDot,inspectText])el.setAttribute('visibility','hidden');});
 svg.append(overlay);
}

async function refreshChartHistory(force=false){
  if(historyPending||(!force&&Date.now()-lastHistoryFetch<4200))return;
  if(!run.model_id){chartTelemetry=[];return;}
  historyPending=true;const version=historyVersion;const modelId=run.model_id;
  try{
    const seconds=Number(get('chartWindow'));
    const result=await api('history',{window_seconds:seconds,max_points:seconds===0||seconds>=86400?1600:800});
    if(version===historyVersion&&modelId===run.model_id){
      chartTelemetry=Array.isArray(result.telemetry)?result.telemetry:[];
      lastHistoryFetch=Date.now();
      plot('monitorLossPlot','loss');plot('monitorAccuracyPlot','accuracy');
    }
  }finally{historyPending=false;}
}
$('chartFocus').addEventListener('change',()=>{const choice=get('chartFocus');$('chartsPanel').classList.toggle('focus-loss',choice==='loss');$('chartsPanel').classList.toggle('focus-accuracy',choice==='accuracy');plot('monitorLossPlot','loss');plot('monitorAccuracyPlot','accuracy');});
$('chartWindow').addEventListener('change',()=>{historyVersion++;lastHistoryFetch=0;chartTelemetry=[];void safe('History',()=>refreshChartHistory(true))();drawRun();});
function drawRun(){const latest=latestTelemetry.at(-1);const active=run.running;const relay=active&&run.relay_only;const label=relay?t('relay'):active?(run.stopping?t('stopping'):t('training')):t('idle');
 setText('runStatus',label+(active&&run.pid?' · PID '+run.pid:''));const loss=latest?.loss;const acc=latest?.accuracy;setText('monitorLoss',loss!=null?Number(loss).toFixed(3):'—');setText('monitorAccuracy',acc!=null?(100*Number(acc)).toFixed(1)+'%':'—');setText('monitorSpeed',latest?.steps_per_s!=null?Number(latest.steps_per_s).toFixed(2):'—');setText('monitorSteps',latest?.steps??0);if(latest?.publish_steps!=null)setText('trainAutoPublish',latest.publish_steps+' / publish');if(latest?.membership_count!=null)setText('trainAutoPeers',String(latest.membership_count)+' online');else if(latest?.channels!=null)setText('trainAutoPeers',String((latest.channels||0)+1)+'+ observed');setText('monitorUpdates',(latest?.emitted??0)+' / '+(latest?.inbound_applied??0)+' updates');setText('monitorQueue',latest?.inbox??'0');setText('monitorChannels',latest?.channels??'0');setText('lossLatest',loss!=null?Number(loss).toFixed(4):'—');setText('accuracyLatest',acc!=null?(Number(acc)*100).toFixed(2)+'%':'—');
 setText('dockModel',active?(selected?.model||get('model')||'KAGAMI'):'NO ACTIVE TRAINING');setText('dockState',label+(latest?.steps!=null?' · '+latest.steps+' steps':''));setText('dockLoss',loss!=null?'LOSS '+Number(loss).toFixed(4):'LOSS —');$('dockProgress').className='progress-fill'+(relay?' relay':active?' running':'');$('dockToggle').textContent=active&&!relay?t('pause'):t('start');
 setText('networkBadge',active?label.toUpperCase():'OFFLINE');setText('networkChannels',latest?.channels??0);setText('networkPush',latest?.push_mib!=null?latest.push_mib+' MiB':'—');setText('networkRepair',latest?.repair_mib!=null?latest.repair_mib+' MiB':'—');setText('networkDuplicates',latest?.inbound_duplicates??'—');setText('networkApplied',latest?.inbound_applied??'—');setText('networkInbox',latest?.inbox??'—');setText('networkLocalName',latest?.peer_id?short(latest.peer_id,25):'—');
 if(currentTab==='training'){plot('monitorLossPlot','loss');plot('monitorAccuracyPlot','accuracy');}drawSelection();}
function drawGraph(){const svg=$('networkGraph');svg.replaceChildren();const last=latestTelemetry.at(-1);const peers=(run.running&&Array.isArray(last?.connected_peers))?last.connected_peers.slice(0,24):[];const actualPeerCount=run.running&&Array.isArray(last?.connected_peers)?last.connected_peers.length:0;const cx=380,cy=210;
 for(let i=0;i<peers.length;i++){const a=(i/peers.length)*2*Math.PI-Math.PI/2;const x=cx+Math.cos(a)*245,y=cy+Math.sin(a)*143;svg.append(svgEl('line',{x1:cx,y1:cy,x2:x,y2:y,stroke:'#376c50','stroke-width':2}));svg.append(svgEl('circle',{cx:x,cy:y,r:26,fill:'#193d2b',stroke:'#379966','stroke-width':2}));svg.append(svgEl('text',{x,y:y+5,'text-anchor':'middle',fill:'#b4f2ce','font-size':14},String(i+1).padStart(2,'0')));svg.append(svgEl('text',{x,y:y+43,'text-anchor':'middle',fill:'#749f85','font-size':10},short(peers[i],12)));}
 svg.append(svgEl('circle',{cx,cy,r:37,fill:'#10b981',stroke:'#b5ffdb','stroke-width':2}));svg.append(svgEl('text',{x:cx,y:cy+5,'text-anchor':'middle',fill:'#082014','font-size':14,'font-weight':'bold'},'LOCAL'));
 if(actualPeerCount>peers.length)svg.append(svgEl('text',{x:cx,y:280,'text-anchor':'middle',fill:'#8cbea0','font-size':11},'+'+(actualPeerCount-peers.length)+' more directly connected peers'));
 if(!peers.length)svg.append(svgEl('text',{x:cx,y:325,'text-anchor':'middle',fill:'#739784','font-size':14},t('noNetwork')));}
let pollPending=false,evalPending=false,historyModel=null;
async function poll(){if(pollPending)return;pollPending=true;try{const result=await api('status');run=result;latestTelemetry=result.telemetry||[];if(result.model_id!==historyModel){historyModel=result.model_id;historyVersion++;chartTelemetry=[];lastHistoryFetch=0;}if(currentTab==='training'&&result.model_id)void refreshChartHistory().catch(()=>{});if(result.checkpoint_ready)setText('storageStatus',result.checkpoint_file);drawRun();drawGraph();const value=result.logs||'—';if(value!==lastLog){const log=$('logs');log.textContent=value.split('\n').slice(-9).join('\n');lastLog=value;}
 }catch(e){setText('bridgeState','Bridge offline · '+e.message);$('bridgeState').classList.remove('online');}finally{pollPending=false;}}
async function loopEval(){if(evalPending)return;evalPending=true;try{await pollEval();}catch(e){setText('evalStatus','ERROR');setText('evalResult','Inference status error: '+String(e?.message||e));}finally{evalPending=false;}}
async function boot(){if(startup)return;startup=true;try{const settings=await api('defaults');for(const id of ['room','model','revision','task','ws','seed','kind'])$(id).value=settings[id];$('modelInitialization').value=settings.initialization;updateWsNotice();for(const [key,id] of Object.entries({reinit_every:'trainReinitEvery',lr:'trainLr',disk_cache_fraction:'trainDiskCacheFraction',wide_lora_rank:'trainWideRank',push_fanout:'trainPushFanout',max_journal_gb:'trainJournalMaxGb'}))$(id).value=settings[key];$('trainStreamingBase').value=settings.streaming_base?'on':'off';sources=settings.sources||[];syncCreationMode(false);drawSources();void refreshAutoPlan().catch(()=>{});
 const dev=await api('device').catch(()=>null);if(dev){setText('deviceLabel',String(dev.device).toUpperCase());setText('hardwareSummary', 'RAM '+(dev.ram_bytes/1073741824).toFixed(1)+' GiB · VRAM '+(dev.vram_bytes/1073741824).toFixed(1)+' GiB');capabilities=dev;}
 $('bridgeState').textContent='● LOCAL BRIDGE';$('bridgeState').classList.add('online');await poll();await refreshModels().catch(()=>{});
 if(settings.last_room){
   try{
     const entry=models.find(x=>x.room_id===settings.last_room&&x.local);
     if(entry){const restored=await api('select',{room:entry.room_id,ws:get('ws')});
       applyManifest(restored,entry.room_id);await reloadMixed(entry.room_id);activatedRoom=entry.room_id;
       selected=entry;drawModels();drawSelection();}
   }catch(e){notify('Resume selection: '+String(e.message||e),true);}
 }
 void loopEval();void updateCheckpoints().catch(()=>{});setInterval(()=>{if(!document.hidden)void poll();},900);setInterval(()=>{if(!document.hidden)void loopEval();},1200);setInterval(()=>{if(!document.hidden&&currentTab==='eval')void updateCheckpoints().catch(()=>{});},5500);
 }catch(e){notify(e.message,true);$('bridgeState').textContent='OFFLINE';startup=false;}}
$('retryBridge').addEventListener('click',()=>{startup=false;void boot();});window.addEventListener('pywebviewready',()=>{void boot();});document.addEventListener('visibilitychange',()=>{if(!document.hidden)void poll();});
translate();void boot();
