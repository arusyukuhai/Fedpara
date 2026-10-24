> **Update 2026-10-23:** Tiny GPT-2 本物の事前学習済みモデル対応については [README_TINY_GPT2_BENCHMARK_2026_10_23.md](README_TINY_GPT2_BENCHMARK_2026_10_23.md) を参照してください。以下は旧 toy モデルの説明です。

# Fedpara — 分散学習有効性ベンチマーク

## 実装範囲（必ず読んでください）

このベンチマークは **PyTorch/UniversalAdapter を用いたローカル・オフライン比較** です。LoHa / LoRA の実コードを使用しますが、Gossip / WebRTC / 実際の分散ネットワークを走らせるものではありません。外部HFモデルを勝手にダウンロードしません。

- 小さな自己回帰 Transformer (causal LM) を固定seedで初期化します。
- 中央集権方式と複数ピア方式で **同一初期重み、同一学習サンプル、同一総トークン数** を使います。
- 各ピアのデータ分割は non-IID。検証用データは学習サンプルと別に生成します。
- 中央集権: 1組のLoHa因子とAdamWを継続学習。
- 分散: 各同期ラウンドで共通ベース重みからピア別にLoHa学習し、**因子ではなく実効 `delta(W)` を平均**してベースに適用します。各ラウンドでローカルAdamWを初期化します（中央方式とoptimizer-stateの継続性が異なる点に注意）。
- LoHa因子初期化を固定seedにし、最初の中央側と各ピアのLoHa因子を同一にします。1ラウンド中の全ピアも同じ初期LoHa因子です。
- 2方式とも同じバッチIDと同じトークン数で学習します（ただしラウンド内の更新順序が異なります）。

## インストールと起動

Fedparaプロジェクトのルートで:

```bash
python -m pip install -r requirements-patches.txt
python -m pip install matplotlib pytest
python -m patchlib.cli benchmark-effectiveness \
  --peers 4 --rounds 12 --local-steps 4 --batch-size 8 \
  --kind loha --rank 8 --threads 1 \
  --output-dir benchmark-results/loha-4
```

結果を分けて保存することで、2, 4, 8ピアの比較が可能です。

```bash
for n in 2 4 8; do
  python -m patchlib.cli benchmark-effectiveness \
    --peers "$n" --rounds 12 --local-steps 4 --batch-size 8 \
    --kind loha --rank 8 --output-dir "benchmark-results/peers-$n"
done
```

**注意:** ピア数を変えると総学習サンプル数も変わります。「同一総トークン数」保証は**同じピア数の中央方式と分散方式のペア比較**に対するものです。ピア数間の正確なスケーリング曲線には総トークン数やデータを固定する別の設計が必要です。

`python -m patchlib.cli gui` でWebView GUIを開き、画面下部の「05 · 分散学習の有効性検証」でも実行できます。GUIは黒背景＋emerald-500 `#10b981` に変更しています。従来の `gui-desktop` (Tkinter) も同系色です。

## 実テキストで動かす場合（CPU・ネットワークオフライン）

8MiB以下のローカル UTF-8 テキスト（例: TinyStories の小さなテキスト抜粋）を用意してください。モデルは字句トークナイザではなく、UTF-8の**バイト**（vocab=256）で学習します。

```bash
python -m patchlib.cli benchmark-effectiveness \
  --peers 4 --rounds 10 --local-steps 4 \
  --vocab 256 --text-file /path/to/stories.txt \
  --output-dir benchmark-results/text-4
```

訓練区間は先頭80%、検証区間は末尾20%です。各ピアへ訓練区間を連続する非重複チャンクとして割り当てます。大量の本番データ・大きなGPUモデルはこのオフライン試験の対象外です。

## 出力

- `effectiveness-result.json` — 全設定、初期/最終Validation Loss、Perplexity、計算秒数、制約事項、trace
- `effectiveness-trace.csv` — ラウンドごとの損失、トークン数、実測した逐次CPU学習秒と補正後秒
- `effectiveness-loss.png` — **同じ総トークン数**を横軸にした損失比較
- `effectiveness-time.png` — 中央側の実測学習CPU時間 vs P2P側の補正学習CPU時間を横軸にした損失比較

## 『計算時間 ÷ ピア数』の意味

`p2p_divided_by_peers_s = sum(all peers' measured training-step CPU seconds) / peer_count`

ご指定の **公平補正指標** です。これは同速ピアが完全並列で動く仮想的な計算秒数であり、**実際のWall Timeではありません**。他に `p2p_max_peer_critical_path_s`（ラウンドごとに最も遅いピアの処理時間を足した値）も保存します。後者は全ピアが並列かつ同期障害なしで動いても下回れない計算段階の目安です。

通信時間、RTT、P2Pプロトコル、モデル転送、参加/退出、モデルの（再）生成・スナップショット化、データロード時間はこのベンチマークの計算時間には**含まれません**。シリアライズしたLoHaパッチのバイト数は参考値として表示しますが、これを実際のネットワークトラフィックと誤解しないでください。

中央方式はoptimizer状態を引き継ぎ、分散方式はラウンドごとにローカルoptimizerを初期化するため、**純粋な通信方式だけの比較ではありません**。これは現在のLoHa分散更新手法の有効性を調べる初期実験であり、実機ベンチマークや収束証明の代わりにはなりません。

## テスト

```bash
python -m pytest -q tests/test_effectiveness_benchmark.py
```
