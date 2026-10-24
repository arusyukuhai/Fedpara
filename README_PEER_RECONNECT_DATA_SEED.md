# Fedpara: SCTP再接続安定化・ピア固有のデータシード（2026-10-22）

前提：`fedpara_datachannel_preopen_fixed.zip` からの差分。Mummyサーバーの変更はありません。過去の `.fedpara-gui`、`*.sqlite3`、`*.identity.pem`、モデルの状態は削除しないでください。

## 接続回復

- `fedpara-patch` DataChannelが閉じたら `link_closed` を立て、WebSocketが開き続けていてもシグナリング側の待機を中断します。古いタスクを終了し、既存の指数バックオフ・ジッター付き再接続に渡します。
- ピア管理ループも `link_closed` を監視し、正常な接続の再取得を最長120秒待たせません。真正の `HandshakeError` は再試行へ格下げせず従来どおり拒否します。
- 署名付きハートビートが多少遅れてもSCTPを即座には切断しません。心拍不達の表示・分母からの除外と物理的な切断は別の状態です。ハートビートが復旧すれば既存のDataChannelを維持します。
- 既定の `--peer-timeout` は18秒から45秒へ変更。WebSocket Ping timeoutも20秒から60秒に延長。長期に署名付き通信が全くない場合の再接続は保ちます。
- ハートビートが受信キューを一時的に満たした際、正しい署名を検証済みの余剰ハートビートは破棄可能にしました。`hello`・`inventory`・`need`・snapshotなどの重要メッセージは破棄しません。

ロギングでは `signed heartbeat delayed` / `signed heartbeat recovered` / `SCTP DataChannel closed; scheduling reconnect` を目印にしてください。`active=1` が短時間表示されるのはあり得ます（本当に届いていない心拍を活発だと偽装しません）が、接続が切れた場合は自動復帰を試みます。

## データのピア別シャッフル

同じルームの `datasets.seed` は共通基準値として保持します。実際のデータ選択は、基準値と**永続的なピアID**から SHA-256 ドメイン分離で63bitのシードを導出します。結果はログの `DATA SAMPLING peer=... seed=...` に表示されます。

- Hugging Faceストリーミング: ピアごとに `shuffle(seed=...)` とデータセット混合用擬似乱数を使用（従来からピアID由来のシードがあり、今回64bit級で明示化）。
- `--text-file`、`--jsonl`: ファイル内容を各ピア固有の順序でエポックごとにシャッフル。
- `--batches-dir`: safetensorsバッチファイルをピア固有の順序でエポックごとにシャッフル。
- 同じピアID・ルーム設定なら再現可能。別ピアは異なる並び。Mummyサーバーへの追加負荷はありません。

**制限:** 独立したシードは同じサンプルを異なるピアが学習しない保証ではありません。また、データセットの読み取り位置を完全に永続化する機能は今回まだ追加していません。再起動すると同じピアは先頭のシャッフル順序から再生します（モデル重み・確定済み学習履歴は以前の保存方式を引き継ぎます）。HFのスナップショット参照を`main`にしている場合、リポジトリ更新で実データが変わる可能性があるため、可能ならコミットハッシュで固定してください。

## 起動

1. 現在の学習・GUIをすべて終了します。
2. 同梱ZIPの内容で既存ソースを更新します。**`.fedpara-gui`やSQLiteは上書きしないでください。**
3. `python -m patchlib.cli gui` を2窓起動し、同じルームへ参加します。
4. ログで各ウィンドウの `DATA SAMPLING` の `peer` と `seed` が違うこと、`living=2`、接続後の `active=2` を確認します。

CLIでは例として `python -m patchlib.cli train-live --help` で `--peer-timeout` を確認できます。

## 確認できた範囲

`python -m pytest tests -q`：**140 passed, 3 skipped**。Pythonコードの構文検査にも成功。切断処理には仮WebSocket/仮DataChannelの統合テストが含まれます。ただし、この環境には `aiortc` がなく、本物の2台間WebRTC実接続やmacOS MPSは未検証です。長時間実機運用の保証はありません。
