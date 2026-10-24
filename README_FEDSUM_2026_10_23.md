# Fedpara: 継続FedSum（v3）

今回のリビジョンは **rank 16・ランダム初期化Frozen TinyStories-1M・Embedding / Position Embedding / LM Headも含むLoHa** を維持し、P2P更新の集約演算だけを **SUM** に変更します。中央集権側の学習処理、optimizer、データ量・検証方法は変更しません。

## 変えたところ

- オフラインベンチマーク: `aggregated[name] += effective_peer_delta` として各ピアの重み差分を**足し合わせます**。以前の `alpha=1./peers` は削除しました。
- ライブWebRTC/Gossip: 独立した署名付きbefore/after LoHa差分イベントを受信順に一度ずつ加算します。`(D(after)-D(before)) / live_denominator` を `(D(after)-D(before))` に変更しました。
- LoHaの因子そのものは足していません。**実効重み差分**を足しています。ローカル因子・AdamW状態の継続、アンカー、署名、イベント重複排除は維持します。
- 新プロトコル: `fedpara-continual-elastic-fedsum-v3`。イベントメタデータの `aggregation=sum` と `denominator=1` を必須とし、旧v2 FedAvgイベントは拒否します。
- 新しいgenesis: `optimizer_protocol=continual-elastic-fedsum-v3`。既存のv2ルーム・SQLite履歴とは互換ではありません。
- `effectiveness-result.json` には `configuration.p2p_aggregation="sum"` と `fairness.peer_aggregation` を記録します。

## 使い方

```bash
python -m patchlib.cli benchmark-effectiveness \
  --model tinystories-1m --initialization random --dataset tinystories \
  --target-scope gpt_neo_full --kind loha --rank 16 --lr 0.0001 \
  --peers 4 --rounds 40 --local-steps 8 --batch-size 4 \
  --sequences 2048 --validation-sequences 1024 \
  --output-dir benchmark-results/fedsum
```

GPU/CPU間の通信時間はオフラインベンチマークに含まれず、`p2p_compute_s_div_peers` は計算時間合計÷ピア数という理想並列推定を維持します。更新差分の割り算をやめたこととは無関係です。

### ライブルーム移行（必須）

1. 既存のv2参加プロセスを停止し、既存の `.fedpara-gui` とjournal/snapshotをバックアップ。
2. **新しいroom ID、新しいjournalパス**でv3を起動。v2のjournalは自動変換しません。
3. 全参加者がv3であることを確認。古いプロセスとイベントを混ぜてはいけません。
4. 旧重みを再利用する必要がある場合は検証済みフルモデルスナップショットを別途エクスポートして、新しいgenesisとして設定。古いroom IDを再利用しないこと。

### 注意事項

4ピアなら、同じ各ピア更新に対してFedAvgの4倍の合成差分になります。学習率を固定するとオーバーシュート/発散しやすくなります。今回は依頼どおり**学習率を自動的に1/4に変更していません**。これは加算の効果を測る実験です。

P2Pライブは非同期Gossipなので、オフライン同期型FedSumと数学的に同じ結果になる保証はありません。また実機WebRTCでの大規模検証は未実施です。
