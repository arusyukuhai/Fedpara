# Fedpara FedSum + 定期LoHa再初期化 (v4)

## 仕様

FedSum（各ピアの実効更新を総和）を維持し、**rank 16** の LoHa 因子を定期的に新しいゼロ更新の因子へ再初期化します。Embedding、Position Embedding、LM Headも引き続き学習対象（共有重みは1組）です。**ランダム初期化のTinyStories-1M構造、Frozen Base + LoHa学習**も維持します。

- **ベンチマーク**: `--reinit-every 20` で20同期ラウンドごと、中央集権側とP2P側ともMerge+再初期化。`0` で従来どおり継続学習。中央集権側は現時点の実効LoHaデルタを一度だけ固定ベースに追加し、新しい因子とOptimizerへ移行します。P2P側はそのラウンドのFedSumデルタが共有ベースに入った後、全ピアの因子/アンカーを更新します。
- **ライブWebRTC**: `--reinit-every 20` で各ピア自身の**20件目、40件目、…の署名付きローカル更新**がSQLiteに確定された時点で当該ピアの因子/アンカーを再初期化。**受信した他ピアの更新は周期に数えません。** レプリカは受信済みの実効差分を一度ずつ足すのみで、遠隔ピアのローカルOptimizer状態は触りません。
- **AdamW**: 再初期化ではモーメントは古い因子座標に対応しているためOptimizerを作り直します。周期内はモーメントが継続します。LoRA-Muonもパラメータの切替に合わせて状態を初期化します。
- **再現性/クラッシュ**: 署名付きイベントメタに `reinit_after`, `reinit_every` を記録。ローカル因子の初期化seedは `genesis:peer_id:seq` のダイジェストから決定し、イベントの再生時に同じ因子へ再初期化。既に確定した重みを失うことはありません。
- **プロトコル**: `fedpara-continual-elastic-fedsum-relora-v4`。genesis に `optimizer_protocol=continual-elastic-fedsum-relora-v4` と `reinit_every` を含めるため、**FedSum v3とのジャーナル・ルーム互換性はありません**。既存ルームはそのままにし、**新しいルームID・ジャーナルで起動**してください。v3のjournalやsnapshotのまま上書きしないでください。

これはReLoRAから着想を得た**周期的Merge/再初期化**で、論文中の細かなoptimizer pruningやwarm restartには完全一致しません。LoHaもReLoRAと同一ではありません。ライブは同期バリアのない非同期Gossipであり、同時に全ピアがリセットするわけではありません。

## 実行例

```bash
# ランダム初期化のTinyStories-1M構造を固定、rank 16 LoHa/FedSumでゼロから学習
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --initialization random --dataset tinystories \
  --target-scope gpt_neo_full --kind loha --rank 16 --lr 0.0001 \
  --peers 4 --rounds 80 --local-steps 8 --batch-size 4 \
  --reinit-every 20 --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/fedsum-relora-20

# 再初期化なし対照群
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --initialization random --dataset tinystories \
  --target-scope gpt_neo_full --kind loha --rank 16 --lr 0.0001 \
  --peers 4 --rounds 80 --local-steps 8 --batch-size 4 \
  --reinit-every 0 --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/fedsum-no-refresh
```

GUIでは学習パネルとベンチマークパネルの両方に「再初期化周期」を追加しました。ベンチマークは40ラウンドを初期値にし、1度以上リセットを含む構成としています。出力JSONには `configuration.reinit_every` と各 `trace[].relora_reinitialized` を書きます。`--reinit-every 0` と同一seed・総トークン数でLossを比較してください。

## 注意点

- 周期を短くすると毎回LoHaがゼロ初期化から学び直すため、必ずしも収束が改善するとは限りません。
- **同期のズレ、非IIDデータ、PeerごとのAdamW状態、通信遅延**は残ります。リセットだけで中央集権に一致する保証はありません。
- ライブ用の署名付き連続イベントの詳細と実モデルの大規模WebRTC動作は未検証です。ローカルテストではイベントリプレイ・無損失Merge・イベント重複拒否・単一プロセスでの疑似学習・全回帰テストを実行しています。
- `--reinit-every`を1など非常に小さくすると学習が進まない可能性があるため、まず20、次に10/40/0の比較を推奨します。
