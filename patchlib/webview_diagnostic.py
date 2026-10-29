"""Minimal pywebview test independent of Fedpara/torch, for macOS focus/bridge diagnosis.

Usage: python -m patchlib.webview_diagnostic
"""
import sys

HTML = '''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.1.10"></script></head>
<body class="min-h-screen bg-[#080e0b] px-6 py-7 font-mono text-sm text-[#e5f6ec]">
<h2 class="mb-4 border-b border-emerald-700 pb-3 text-xl font-bold text-emerald-300">Fedpara WebView 操作診断</h2>
<p class="my-3 text-[#91a79b]">クリック・入力・Pythonブリッジを独立して確認します。</p>
<button id="click" class="my-2 cursor-pointer border border-emerald-500 bg-emerald-500 px-4 py-2 font-bold text-emerald-950 hover:bg-emerald-300">1. JavaScriptクリックテスト</button>
<div id="local" class="min-h-8 border border-emerald-900 bg-[#0d2116] p-3">待機中</div>
<input id="input" class="my-3 w-full border border-emerald-700 bg-[#0c1912] px-3 py-2.5 text-emerald-50 caret-emerald-400 focus:outline-2 focus:outline-emerald-400" placeholder="2. ここに文字を入力できますか？">
<button id="python" class="my-2 cursor-pointer border border-emerald-500 bg-emerald-500 px-4 py-2 font-bold text-emerald-950 hover:bg-emerald-300">3. Pythonへの接続テスト</button>
<div id="remote" class="min-h-8 border border-emerald-900 bg-[#0d2116] p-3">待機中</div>
<script>
const local=document.getElementById('local'), remote=document.getElementById('remote');
let clicks=0;
document.getElementById('click').onclick=()=>{local.textContent='クリック成功 '+(++clicks)+'回';};
document.getElementById('python').onclick=async()=>{
 remote.textContent='Pythonへの接続を確認中…';
 if(!window.pywebview?.api?.ping){remote.textContent='Pythonブリッジ未接続';return;}
 try {remote.textContent=JSON.stringify(await window.pywebview.api.ping());}
 catch(e){remote.textContent='Python応答エラー: '+String(e);}
};
window.addEventListener('pywebviewready',()=>{remote.textContent='Pythonブリッジの準備完了';});
</script></body></html>'''

class Api:
    def ping(self):
        return {'ok': True, 'python': sys.executable}

def main():
    import webview
    print('Python executable:', sys.executable, flush=True)
    print('This minimal diagnostic has no TCP listener and does not load torch.', flush=True)
    webview.create_window('Fedpara WebView Diagnostic', html=HTML, js_api=Api(),
                          width=620, height=470, resizable=True)
    webview.start(debug=True)

if __name__ == '__main__':
    main()
