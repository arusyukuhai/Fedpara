> **Random frozen-base LoHa is now the default.** See [README_TINYSTORIES_1M_RANDOM_LOHA_2026_10_23.md](README_TINYSTORIES_1M_RANDOM_LOHA_2026_10_23.md). The instructions below describe the older pretrained-weight behavior; add `--initialization pretrained` to reproduce them.

**学習・検証データ増量版:** [README_TINYSTORIES_DATA_VOLUME_2026_10_23.md](README_TINYSTORIES_DATA_VOLUME_2026_10_23.md)（学習2,048系列/ピア、検証1,024系列。CLI・GUIから変更可能）。

# Fedpara benchmark: TinyStories-1M (GPT-Neo, continual FedAvg)

## 重要

`sshleifer/tiny-gpt2` は幅2の機能確認用のチェックポイントでした。ベンチマークのモデルを公式 `roneneldan/TinyStories-1M` に変更しました。GPT-Neo 8層 / hidden size 64、GPT-2 BPEトークナイザです。

- Model: `roneneldan/TinyStories-1M`
- Revision (固定): `77f1b168e219585646439073245fe87e56b3023e`
- Weights: **pytorch_model.bin ONLY** (no safetensors). Safe loader: PyTorch >=2.6, Transformers `weights_only=True`, `trust_remote_code=False`. `torch.load(weights_only=False)` は使いません。
- Default rank 8 / context 64 / lr 3e-4, peers 4.
- Dataset: `roneneldan/TinyStories` のtrain/validationを別々にstreaming。データセットrevision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` を固定。
- モデルは既にTinyStoriesで事前学習されています。初期Lossが低く改善幅が小さいのは必ずしも障害ではありません。継続Fine-tuningの比較です。
- 全学習経路でLoHa/AdamWを保持し、学習前後の **実効重み差分**を平均。計算時間÷ピア数は理想的な推定値、実通信速度ではありません。
- ライブP2P GUIのモデル既定値は変更しません。変更対象は**ベンチマークのみ**です。

## 準備

```bash
python -m pip install -U 'torch>=2.6' 'transformers>=4.45' 'huggingface_hub>=0.25' 'datasets>=4,<5' matplotlib
```

## 実行

```bash
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --peers 4 --rounds 12 --local-steps 2 \
  --batch-size 2 --rank 8 --context 64 --sequences 64 \
  --lr 0.0003 --output-dir benchmark-results/tinystories-1m
```

初回ダウンロード後のスモークテスト:

```bash
python -m patchlib.cli benchmark-effectiveness --model tinystories-1m \
  --dataset builtin --peers 2 --rounds 1 --local-steps 1 \
  --batch-size 1 --rank 8 --context 32 --sequences 16 \
  --output-dir benchmark-results/tinystories-1m-smoke
```

GUI:`python -m patchlib.cli gui` → 「05 · 分散学習の有効性検証」。既定モデルはTinyStories-1M、既定データはTinyStories。

## 検証方法と制約

`tests/test_tinystories_1m_effectiveness.py` に安全な重みロードとGPT-Neoモデルの受け渡しのテストを追加しました。リモートのチェックポイントを実際にダウンロードするE2Eはこの環境で未実施です。CPU上で4+ピアの大きなモデルを複製するためRAMと時間を消費します。

ベンチマークはオフライン同期型であり、P2P/WebRTC通信、Gossip遅延、NAT越えを評価しません。
