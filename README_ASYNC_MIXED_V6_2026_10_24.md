# Fedpara v6 — 非同期受信パイプライン + LoHa/LoRA混合分解

この版は **fedpara_live_gui_fedsum_relora_v5_full.zip** からの更新です。以前のライブ更新イベント／room genesisとは非互換です。**必ず新しいルーム・ジャーナルを使用してください。**

## 変更

* WebRTC / aiortcの受信イベントを、学習のforward/backward中も別のasyncioタスクで取り込むよう変更。従来の「ローカル学習1ステップ終了後に受信キューをポーリング」を廃止。
* 非同期受信の際、`seal_if_needed()` を呼ばない。各ピアは設定されたローカルステップ（既定4）で公開し、受信に関係なくLoHa/LoRAとAdamWの状態を継続。
* 署名・テンソル検査を `asyncio.to_thread` に移し、受信イベント専用の有界FIFOキューを追加（`--inbound-queue`、既定32）。キューが埋まった場合は既存のRTC受信キューに保留し、溢れたものは署名付きinventory/needで再転送。メモリを無制限に割り当てる実装ではありません。
* SQLite追記とモデル更新は**単一の非同期コミットロック**で順序管理。SQLiteは元のイベントループスレッドで操作し、行列更新は別スレッド + adapter lockで守る。署名検証失敗やジャーナル障害は停止して明示的に通知。
* 新規参加の署名付きsnapshot境界と、受信更新のコミットは排他制御。スナップショット転送時には整合性維持のため短時間の待機が残ります。
* **GPT-NeoのToken Embedding・Position Embedding・共有LM Headはrank 4 LoRA**。Attention/MLPの学習対象はrank 16 LoHaのまま。LM HeadとToken Embeddingが同じParameterなら1アダプターのみを共有。
* 全ターゲットの個別 `kind,rank,alpha` をレイアウトハッシュと署名付きイベントのメタデータに格納。v5以前の署名付き差分はv6として解釈できません。
* `--durable-every` 既定を4に変更。毎ステップのローカル作業状態のsafetensors書き出し待ちを減らす。**クラッシュ時に最大3ローカル学習ステップの未公開作業が失われ得る**。公開済み更新は署名付きSQLiteイベントジャーナルに残ります。最小ロス優先なら `--durable-every 1` に変更してください。
* GUI初期値: LoHa16 / wide LoRA4 / LR1e-4 / local_steps4 / reinit20 / 想定ピア5 / 公開イベント上限200。

## GUIの起動

```sh
python -m pip install -r requirements-patches.txt
python -m pip install 'pywebview>=5,<7'
python -m patchlib.cli gui
```

ルームを新規作成し、公開イベント上限200で開始します。ルーム既定モデルはランダム初期化した `roneneldan/TinyStories-1M` とTinyStories学習データです。

**同期シミュレーションの `--rounds 200` とライブの200イベントは同じ意味ではありません。** ライブFedSumは各ピアのイベントを独立に適用し、同期バリアがありません。`--expected-peers 5` は想定規模の記録値であり、5人が揃うまで学習を待つ設定ではありません。ライブ側 `reinit-every 20` は各ピアの20公開ごと、シミュレーションは20ラウンドごとです。

## 通信量の目安

Embeddingサイズが50,257×64、rank16の純LoHaでは4因子、更新前後を含めて約25.8MB/イベント。rank4 LoRAへ切り替えると同一行列の更新前後で**約1.61MB/イベント**、約16分の1。実際のwireには他の層・署名・メタデータが加わります。通信で送るのは before/after 因子であり、`float32` のままです。

## 検証と限界

```sh
PYTHONPATH=. python -m pytest -q tests/test_async_mixed_v6.py tests/test_elastic.py tests/test_live_gui_random_tinystories.py
```

非同期受信は **ローカル学習とネットワーク処理を重ねられる設計**ですが、**完全ロックフリーではありません**。同じモデルの安全なforward/backward、Optimizer step、遠隔更新適用、Snapshot exportが競合する瞬間には、`adapter.lock` 等で順序を守ります。CPU/ディスク帯域が飽和している場合に必ず速度向上するとは限りません。

5ピアの実WebRTC/aiortc E2E、NAT越え、通信量と収束精度の実機評価はこの環境では未実施です。`aiortc`がこの環境にインストールされていないためです。専用テストでは共有Head、更新の重複排除、署名付きジャーナル再生、LoHa因子・AdamWの継続性、GUI引数の接続などを検証しています。
