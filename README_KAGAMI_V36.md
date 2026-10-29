# Kagami v36 — checkpoint save, trained EVAL, long-range history, typography

v35 の保存・EVAL・グラフ履歴に関する不具合を修正した版です。

## 1. チェックポイント保存の改善

- TRAINING の「学習停止（中継継続）」で、一時停止フラグを立てると同時に学習済み実効重みの保存を要求します。WebRTC Gossip と中継は停止しません。
- EVAL の「Save current weights」からも保存を要求できます。
- 学習済み重みが未保存で EVAL の「Generate」を押すと、学習ワーカーが生きている場合は自動的に保存を要求し、成功後に推論を開始します。**初期重みへ黙ってフォールバックしません**。
- 保存は重みごとのシャードを生成し、manifest を最後にアトミックに公開します。保存中の状態・エラーは `eval-snapshot-<profile>.status.json` へ書き、EVAL パネルには `SAVING` / `SAVE ERROR` / `SAVED CHECKPOINT` などを表示します。保存処理の例外が学習・中継ループを無条件に殺さないよう分離しています。
- 保存先は `.fedpara-gui/rooms/<model-id>/checkpoints/<profile>/effective-*.shards.json`、重みオブジェクトは対応する `.fedpara-gui/rooms/<model-id>/models/<profile>/` にあります。既存の保存先を開くボタンも使用できます。
- 停止したワーカーの未保存RAM内の重みは、あとから復元できません。**学習中または中継中に保存**してください。起動できない場合はTRAININGログを確認してください。

## 2. グラフの全期間表示（根本原因の修正）

- `recent_samples(..., max_samples=1)` が、JSONLの最後の空要素だけを選ぶバグを修正。これが全期間APIが常に空になっていた直接原因で、GUIが直近320件にフォールバックしていました。
- `history_samples` の時間範囲・統計項目を従来どおり維持しつつ、最大1600点まで分散サンプリングします。**点数上限と表示できるステップ範囲は別です**。
- 実行中のローカルプロファイルの履歴を `.fedpara-gui/telemetry/history-<room>-<profile>.jsonl` に追記して保持します。再起動間でも `steps` と `elapsed_s` は単調増加します。モデルを選択した場合、以前のローカル履歴も表示します。
- 過去のv35で保存されていなかったサンプルは復元できません。旧バージョンのランダムな telemetry JSONL は上書き・削除しませんが、新しい連結履歴には自動的にマージされません。

## 3. 合成フォント／CDN

- 欧文: 利用可能であれば `Myriad Pro SemiCondensed` を優先、次にユーザー指定の `https://fonts.cdnfonts.com/css/myriad-pro` が提供する `Myriad Pro` を使用。
- 和文: Google Fonts `Noto Sans JP` を使用。
- フォントのCDN指定はページ描画後に非同期追加する方式です。CDN待ちでアプリ起動が停止しません。OFFLINEではシステムフォントにフォールバックします。CDN側が SemiCondensed を提供していない場合に Myriad Pro 通常幅となる可能性があります。
- CSPの変更はフォント用のスタイル2ドメイン／フォントファイル配信2ドメインの許可に限定。外部 JavaScript の読み込みは許可していません。フォントバイナリの同梱はしていません。
- CourierはデスクトップWebView・旧Tkinterテーマの指定から取り除きました。角丸は0のままです。

## 4. 検証と限界

- `python -m pytest -q tests/test_kagami_v36.py` が成功。
- 新旧関連70件のテストが成功（v31/v32/v34、shards、offload、telemetry）。
- Chromium 1366x768 / 1024x660: JavaScriptエラーなし、画面全体のスクロールなし、両フォントCDNへのリクエストを確認。CDNの配信自体は実行環境がインターネット遮断のため未確認。
- 大規模Transformers/GPU環境と公開インターネット経由のWebRTC多ピア実機テストは未実施。保存容量やモデル固有のdtype・独自アーキテクチャの制約は残ります。

## テスト

```
python -m pytest -q tests/test_kagami_v36.py tests/test_kagami_v31.py tests/test_kagami_v32.py tests/test_kagami_v34.py tests/test_local_telemetry.py tests/test_live_snapshot.py tests/test_shards.py tests/test_shard_runtime_v27.py tests/test_offload_v28.py
```
