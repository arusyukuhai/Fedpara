# 実験A 改訂版：Full FT vs 継続FedAvg（LoHa）

前回の `--lr 0.0003` 同一設定は、事前学習済み TinyStories-1M の Full FT 側に過大な更新を与え得ます。本バージョンでは**中央Full FT**と**分散LoHa**の学習率を分離し、初期Loss一致を実測して不一致なら実験を停止します。中央LoHaは復活させていません。

## 実環境での再実験（TinyStories-1M）

```bash
python -m pip install -r requirements-patches.txt
python -m pip install -U 'torch>=2.6' 'transformers>=4.45' 'datasets>=4,<5' matplotlib
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --dataset tinystories \
  --peers 4 --rounds 40 --local-steps 4 --batch-size 2 \
  --sequences 128 --context 64 --rank 8 --kind loha \
  --central-lr 0.00001 --peer-lr 0.0003 --max-grad-norm 1.0 \
  --seed 20261023 --output-dir benchmark-results/experiment-a/seed20261023
```

`--lr` は旧互換で両方に同じ値を設定してしまうため、**実験Aには使わないこと**。`--central-lr` `--peer-lr` を使います。GUIにも独立した入力欄を追加しました。

学習開始前のCEが `BASELINE tokens=0...` に表示され、両枝の初期CEが 1e-5 の範囲で一致しなければ `FAIRNESS FAIL` で停止します。Lossと時間のグラフに「0トークン」の初期点を追加。Full FT/LoHaの学習・検証Loss、最終未使用テスト集合のCE、データハッシュ、学習可能パラメータ数、所要時間をJSON/CSVに出力します。公式TinyStoriesの `validation` ストーリーから互いに異なる窓を選び、前半を選択用validation、後半を最終testに割り当てます。学習は元のtrain splitのままです。

計算時間 `p2p_total_compute_s / peers` は**計算時間のみ**の理想値。加えてラウンドごとの最遅ピア時間合計 `p2p_max_peer_s_sum`、中央フルFT実測時間を記録します。どちらも実WebRTC通信時間を含みません。

## 複数seedおよび学習率探索（オプション）

```bash
python scripts/run_experiment_a.py \
  --central-lrs 0.000003,0.00001,0.00003 \
  --peer-lrs 0.0001,0.0003,0.001 \
  --seeds 20261023,20261024,20261025 \
  --rounds 40 --local-steps 4 --peers 4 --batch-size 2 \
  --sequences 128 --rank 8 --context 64 \
  --output-dir benchmark-results/experiment-a-grid
```

この例は **27 回の学習**を行うため処理時間・メモリ・データ通信に注意してください。まずは上の単一実行を推奨します。事前の実行数確認は `--dry-run` が使えます。`experiment-a-runs.csv` に全実験、`experiment-a-grid.csv` に平均・標準偏差が出ます。学習率候補が2組以上なら、探索用の全試行に `--skip-test` を自動で渡してテストCEを非公開にします。**validation CEのみ**を使いFull FTとLoHaの学習率を確定したら、選んだ組み合わせで上の単一実行CLIを再実行し、最終test CEを測ってください。最終テストを使って学習率を再調整したら、未知の新しいホールドアウトが必要になります。

## 公平さの限界

* 初期重みと学習サンプル・総トークン数は一致（固定モデル・固定データリビジョン）。ただし学習パラメータ数はFull FTとLoHaで大幅に異なります。
* Full FTは継続AdamW、LoHaは各ピアで継続AdamW＋実効更新のFedAvgです。データ分割、局所更新、集約機構は方式ごとに異なるため、これは「システムA vs システムB」の比較です。
* 分散のGPUが並列に動いているわけではなく、オフラインで各ピアを順番に学習させているシミュレーションです。**通信を考慮した高速化の証明ではありません。**
* LoHaモデルは少ない更新可能パラメータで同程度のCEを達成するなら意義がありますが、Full FTを故意に不利にした学習率で結果を解釈してはいけません。
* 今回の環境には本物のHFモデルがキャッシュされずTransformers/datasetsも未導入のため、実TinyStories-1M E2Eはユーザー環境で再実行してください。toyモデル、模擬HFモデルの回帰テストで動作経路を検証しています。
