# Fedpara WebView — クリックできない・フリーズしたように見える場合

修正版の上部には「GUI動作診断」があります。

1. 「クリックテスト」を押して回数が増えるか確認します。このボタンはPython APIを呼びません。
2. 回数が増えるが「Pythonブリッジ未接続」なら、WebView自体は操作できています。`FEDPARA_GUI_DEBUG=1 python -m patchlib.cli gui` でログを表示してください。
3. 回数が増えず、入力欄にも文字が入らないなら、GUIウィンドウのフォーカスやmacOSのWebView実装を疑います。
4. `python -m patchlib.webview_diagnostic` を実行すると、Fedpara/torchに依存しない小さな診断ウィンドウが開きます。これも押せなければアプリ固有の学習機能が原因ではない可能性が高いです。
5. macOSでは、OS付属Pythonから作った仮想環境のWebViewでキーボードフォーカス問題が報告されています。Homebrew等のPythonで新しい仮想環境を作り直すと改善する場合があります。

例（Homebrew Pythonを使用。既存の環境を変更しません）：

```bash
brew install python
"$(brew --prefix)/bin/python3" -m venv .venv-webview
source .venv-webview/bin/activate
python -m pip install -r requirements-patches.txt -r requirements-gui.txt
FEDPARA_GUI_DEBUG=1 python -m patchlib.cli gui
```

UI上で明示的に「クリック正常」と表示されても、Pythonへの接続と学習の実行が成功したことにはなりません。

## 変更点

- UI操作そのものとPythonブリッジ状態を別表示
- 接続ができない場合に4.5秒で診断メッセージを表示して再試行可能
- Python APIメソッドが一部欠けても他の操作を待ち続けない
- Pythonとは無関係のクリックテストを画面上部に配置
- `FEDPARA_GUI_DEBUG=1` でpywebviewのデバッグを有効化
- 独立した `patchlib.webview_diagnostic` を追加

## 制約

macOS実機のフォーカス問題やWebKit上のネイティブ表示はこの開発環境から直接検証できません。
