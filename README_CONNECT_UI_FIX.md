# Fedpara: ICE接続ループとWebView応答性の修正（2026-10-21）

## 修正対象

- `patchlib/elasticnet.py`: 接続状態とロールを記録。ICE接続待ちを30～36秒で強制リセットしていた処理を120秒のタイムアウトに変更。リトライに方向別ジッターを設け、再接続の連鎖を抑制。安定した接続ではリトライ履歴をリセット。
- `patchlib/elasticnet.py`: ハートビートを学習ループから独立したasyncタスクに分離。大きいチェックポイントの非同期書き出し中でも既存DataChannelの生存を維持する。
- `patchlib/p2p.py`: Offer/Answer/ICE/DTLSの段階と状態をログへ追加。シグナリング接続が失われても既存DataChannelが生きているなら維持する。Offerの二重送信を防止。
- `web/desktop_webview.js`: 状態ポーリングを単一の実行中要求に制限、3.5秒へ間引き、変化しないログの再描画を中止。Pythonブリッジ呼び出しにGUI側の期限を設定（バックエンド処理を強制キャンセルするものではない）。
- `patchlib/gui.py`: GUI起動の学習サブプロセスの数値計算スレッドを控えめに設定し、OSスケジューラで学習側の優先度を下げる（可能な場合）。`FEDPARA_WORKER_THREADS=N`で上書き可能。
- `patchlib/webview_gui.py`: ウィンドウがフォーカス可能なことを明示。

## 実行と診断

現行バージョンの `src/main.nim` でMummyを再起動し、古いGUI/学習プロセスは終了してください。`.fedpara-gui`、SQLite、鍵、シャード、学習チェックポイントは**削除しないでください**。

```
python scripts/check_group_presence.py
python scripts/check_live_p2p.py
FEDPARA_GUI_DEBUG=1 python -m patchlib.cli gui
```

各ウィンドウで同一ルームIDを使い、学習ログの `P2P ... phase=... role=... ice=... rtc=...` と `JOIN WAIT` を確認してください。

- `living=2 active=1`: 2者入室は認識しているが、認証済みの接続はまだない。`phase=waiting-peer`が続けばpair roomのシグナリングを、`phase=gathering-offer`が続けばSTUN/ICEを、`rtc=failed`ならICE接続候補・TURN設定を検査。
- `living=1`: ルーム入室・ピアID・シグナリング接続を確認。
- `active=2 synced=1`: DataChannel認証は成功し、更新履歴・最新重み同期が未完了。`SNAPSHOT OFFER/REQUEST/READY`ログを確認。

**重要**: `active`は自身を含む人数です。2台なら `active=2` が接続完了の目安。最新重みの同期も終わるまで新規ピアは学習しません。

## 制約

macOS + aiortc + Mummy + TURNの実接続検証と、macOSネイティブWebViewのフォーカス動作確認はこの作業環境では実施できません。特にネットワークでICE候補が到達できない場合、リトライタイミング改善だけでは接続できません。`scripts/check_live_p2p.py` の結果と両方のログが次の切り分けに必要です。

pywebview公式FAQには、macOS同梱Pythonで作成した仮想環境のウィンドウがキーボードのフォーカスを失う既知事象が記載されています。その症状の場合はHomebrew Pythonで仮想環境を作る方法を検討してください（既存の学習保存先は消さずに）。
