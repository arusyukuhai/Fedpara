# Fedpara ライブP2P / GUI統合 — Random TinyStories-1M + FedSum + Periodic ReLoRA

## このリビジョンの対象

従来の **オフライン・有効性ベンチマーク** (`benchmark-effectiveness`) だけでなく、メインWebView GUI (`python -m patchlib.cli gui`) の **「04 · P2P 継続FedSum学習」** から起動する **実際のWebRTC署名付きイベント学習** にも、モデル初期化・学習設定を渡します。v4のFedSum/ReLoRA更新エンジンは保持しています。

### ライブルームの新しいデフォルト

| 設定 | 既定値 | 備考 |
|---|---|---|
| Model | `roneneldan/TinyStories-1M` | GPT-Neoの公式アーキテクチャ |
| Model revision | `77f1b168e219585646439073245fe87e56b3023e` | 固定コミット |
| **Model initialization** | **random** | 公式configから構築、事前学習済み重みを取得せず、本体のParameterは学習せずLoHaのみを訓練 |
| Initialization seed | 42 (ルームdataset seedと共用) | ルームmanifestに固定 |
| Dataset | `roneneldan/TinyStories` train | 固定commit `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| Adapter | LoHa | gpt_neo_full, `balanced`初期化 |
| Target modules | Attention+MLP Linear, Word/Position Embedding, LM Head | Embedding/Headの共有Parameterには単一のOverlayを登録 |
| Rank | 16 | GUIで変更可能、**新規ルームが必要** |
| Aggregation | FedSum | `W += ΣΔW_i`、ピア数除算なし |
| Optimizer | AdamW | アダプター因子の状態を継続 |
| LR | 1e-4 | GUIから変更可。ライブは定数LR（オフラインベンチのwarmup+cosineとは異なる） |
| Local steps | 64 (自動調整で変化することあり) | ネットワーク更新ごとのローカルstep数 |
| Context | 64 | GUI `max_length` |
| LoHa reinit | **20 ローカル署名付き更新** | 因子+AdamWリセット、マージ済みベースは維持 |

### ルームGenesisについて

- 新規ランダムルームは `genesis.random.fstate` をローカルに作成・ロック・検証します。**以前の `genesis.fstate` は再利用しません**。
- seedとmodel/revisionを固定し、起動時にCPU上で同一configから独立生成します。既存Peerのスナップショットと署名付きイベントを取得・検証したうえで同期します。
- `manifest.model.description` が `fedpara:random-tinystories-1m:v1` のときのみrandom経路を有効化。ルームmanifest変更を拒否するため、異なるモデル初期化方式の混入を防ぎます。
- 旧FedAvg・旧FedSum・異なるrank／周期のSQLiteイベント履歴とは混ぜないでください。**新しいルームID**で開始してください。
- 新規ランダムルームは公式config/tokenizerの初回取得にインターネット接続が必要です（**モデルの学習済み重みは不要**）。
- 参加者がカスタム `*.fstate` を明示指定した場合は、そのTensor内容を検証し、同一モデルのGenesisチェックポイントとして扱います。自動ランダムGenesisそのものと同じとは限りません。

### 実行

```bash
python -m pip install -r requirements-patches.txt
python -m pip install 'pywebview>=5,<7'
python -m patchlib.cli gui
```

GUIで新しいルームを作り「ルーム設定を保存」→「学習開始」。シグナリングサーバーは従来どおり別途起動が必要です。

CLIから同等のランダムモデルを使用する場合:

```bash
python -m patchlib.cli train-live \
  --room-id <NEW_ROOM_ID> --model roneneldan/TinyStories-1M \
  --revision 77f1b168e219585646439073245fe87e56b3023e \
  --initialization random --initial-seed 42 \
  --kind loha --rank 16 --target-scope gpt_neo_full --init-style balanced \
  --reinit-every 20 --optimizer adamw --lr 0.0001 \
  --room-manifest <ROOM_MANIFEST_PATH> --state-base <NEW_GENESIS_PATH> \
  --auto-state --journal <NEW_JOURNAL_PATH> --ws ws://127.0.0.1:8080/ws
```

**比較上の注意:** ライブ学習はWebRTC/Gossipによる非同期更新です。以前のグラフは通信のない同期FedSumシミュレーションです。オフライン `reinit-every=20` は20ラウンドごと、ライブ側では各ピアごとの20 **ローカル公開更新**ごとに再初期化されます。この2つは同じ計算量・時間を表しません。またライブの学習率は定数で、ベンチマークで用いたwarmup/cosineのスケジュールは実装していません。

## 検証

```bash
python -m pytest -q tests/test_live_gui_random_tinystories.py \
  tests/test_gui.py tests/test_room_scaling_directory.py \
  tests/test_webview_gui.py tests/test_relora_periodic_refresh.py \
  tests/test_loha_rank16_initialization.py tests/test_continual_elastic.py
```

「configだけから決定的初期化」「学習済み重みを取得しない」「Embedding/Head共有・原重み凍結」「同じGenesisを再利用」「GUI→実学習コマンド」「異なる初期化方式のルーム書き換え拒否」をローカル・ネット不要のスタブモデルで確認します。Hugging Faceの実チェックポイントをオンライン取得し、本物の5端末WebRTCで学習完走するE2E試験は未実施です。
