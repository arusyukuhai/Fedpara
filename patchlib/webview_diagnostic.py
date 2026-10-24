"""Minimal pywebview test independent of Fedpara/torch, for macOS focus/bridge diagnosis.

Usage: python -m patchlib.webview_diagnostic
"""
import sys

HTML = '''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<style>body{font:16px system-ui;padding:25px;background:#15223a;color:white}
button,input{font:inherit;padding:10px;margin:8px 0}button{cursor:pointer}
#local,#remote{padding:8px;background:#293d5d;min-height:25px}</style></head>
<body><h2>Fedpara WebView 操作診断</h2>
<p>クリック・入力・Pythonブリッジを独立して確認します。</p>
<button id="click">1. JavaScriptクリックテスト</button><div id="local">待機中</div>
<input id="input" placeholder="2. ここに文字を入力できますか？">
<button id="python">3. Pythonへの接続テスト</button><div id="remote">待機中</div>
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
