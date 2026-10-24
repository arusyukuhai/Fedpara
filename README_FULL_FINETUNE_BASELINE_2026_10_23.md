# Fedpara: 中央集権 Full Fine-Tuning vs 継続FedAvg（LoHa / LoRA）

## 変更点

ベンチマークから **中央集権LoHaを削除**しました。中央側は同じ初期チェックポイントから `torch.optim.AdamW(model.parameters(), lr=...)` によるフルファインチューニングを行い、Embedding・Attention・MLP・LayerNorm・LM Head等の更新可能パラメータをすべて学習します。分散側は従来の継続LoHa/LoRA＋実効重み差分FedAvgを使い、毎ラウンドのLoHa因子とAdamW状態を保持します。**ライブP2P学習そのもののプロトコルは変更しません。**

同一seed / 初期チェックポイント / データ・バッチ系列 / 総学習トークン数です。中央側はパラメータ全体を学習し、分散側はアダプタだけを学習するため、**更新可能パラメータ数、容量、オプティマイザの次元は公平ではありません**。この実験は「より大きな学習容量を許した中央集権の基準」と「通信軽量な分散PEFT」の比較であり、純粋なFedAvgの優劣を分離するものではありません。結果JSONに両方式の学習可能パラメータ数を記載します。

Toyモデルの評価時に、学習していないEmbedding等がランダムに再生成される問題も修正しました。中央・分散双方は同じ初期モデル重みから評価します。

## 実行例（TinyStories-1M・公式TinyStoriesデータ）

```bash
python -m pip install -r requirements-patches.txt
python -m pip install -U "torch>=2.6" "transformers>=4.45" "datasets>=4,<5" matplotlib
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --peers 4 --rounds 12 --local-steps 2 --batch-size 2 \
  --rank 8 --kind loha --context 64 --sequences 64 \
  --lr 0.0003 --output-dir benchmark-results/full-vs-fedavg
```

GUI: `python -m patchlib.cli gui` → `05 · 分散学習の有効性検証`。

## 出力と計算時間

- `effectiveness-loss.png` は総トークン数に対する中央 Full FT / 分散継続LoHa のValidation CE。
- `effectiveness-time.png` は中央の**学習ステップ実測計算秒**と、分散の**各ピアの学習ステップ実測秒の合計÷ピア数**による理想並列比較。
- `effectiveness-result.json` に実験条件・学習可能パラメータ数・CE/PPL・計算秒・注意点・履歴。
- `effectiveness-trace.csv` にラウンドごとの総学習トークン数、双方のValidation CE・実際の最適化ステップで観測した平均Training CE、計算秒。

分散の計算秒÷ピア数は、計算能力が同じピアが完全並列に動く仮定の値です。通信、同期、モデルロード、評価、ラウンド集約にかかる時間を含まないため、現実の高速化を保証しません。`p2p_max_peer_critical_path_s` はラウンドごとの最遅ピアを合算した別の理想化推定です。

`--model toy` で外部モデルをダウンロードしないローカル動作試験もできます。TinyStories-1Mは既にTinyStoriesで事前学習済みであり、フルFT側の検証Lossが悪化する場合は、出力の`central_step_train_loss`と比較して過学習・不安定化を診断してください。これは各更新ステップの平均損失であり、ラウンド末時点の全訓練集合Lossではありません。ベンチマークは **オフライン同期シミュレーション**、実WebRTC/Gossipは未測定です。
