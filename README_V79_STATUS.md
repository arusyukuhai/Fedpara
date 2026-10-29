# Kagami v79 — Actor世代差に対するPPO Criticの軌跡重要度補正

## 何を実装したか

- v78の分散Critic（署名付き価値ヘッド差分/世代証明/復旧）を維持。
- PPO rolloutで既に記録していた behavior Actor の各生成トークンの `old_lp` を使い、
  同じ軌跡を現Actorに再入力して全トークンの `target_lp` を計算する。
- **状態分布の変化と、将来の行動分布の変化の両方**を反映する、非クリップの軌跡重要度サンプリング（IS）による価値損失を `patchlib/offpolicy_critic_v79.py` に追加。
- v79の分散Critic新規ルームでは `critic_offpolicy=auto` が `trajectory_is` となり、共有 objective policyに `critic_offpolicy=unclipped-trajectory-is-mc-v79` をピン。旧v78ルームでは明示的に `--critic-offpolicy gae` で旧設定を使う。旧ルームの自動変更はしない。
- CLIは `--critic-offpolicy {auto,gae,trajectory_is}`、GUIは TRAINING → VALUE CRITIC → Off-policy value correction から操作可能。
- ActorのPPO clipped surrogate/GAE advantageは変更せず、**Critic学習の損失だけ**を補正。
- 数値オーバーフロー、NaN、shape不一致はfail-closed。重要度比を安易にclip/normalizeしない。

## 補正式と証明対象

behavior policyを μ、現在の target Actorを π、T個の生成トークンを a_0..a_(T-1)、終端のみの固定報酬を R、gammaを γ とする。

```
r_t = π(a_t|s_t) / μ(a_t|s_t)
W_<t = product_(j<t) r_j       (state occupancy ratio)
W = product_(j<T) r_j          (whole-trajectory ratio)
G_t = γ^(T-1-t) R

L_hat = (1/(2 T)) sum_t [W_<t Vφ(s_t)^2 - 2 W Vφ(s_t) G_t + W G_t^2]
```

behavior policyでサンプルした全軌跡上の `L_hat` の期待値は、条件が満たされれば現Actor π の `E[0.5 * mean_t (Vφ(s_t)-G_t)^2]` と一致する（そのためCritic勾配も期待値で一致）。
`W_<t` は行動履歴による状態到達確率を、`W` は報酬の項における全軌跡の分布差を補正する。
全列挙の二段二値自動回帰モデルで、最初の生成行動に依存するCritic状態と報酬を使って期待勾配の一致を確認した。

## 正しさの必要条件と限界（重要）

1. 行動空間・Tokenizer・prompt分布・環境遷移・Reward定義がbehavior/targetで同一、behaviorがtargetの全行動をカバーすること。
2. behaviorの各選択行動の対数確率が**生成時点に記録**されていること。予測だけ/古いcriticのパラメータだけでは復元できない。
3. 対象Actorの方策は一回の勾配計算中固定、importance ratiosの勾配をCritic損失からActorへ戻さないこと。
4. **期待値として無バイアス**という意味であって、有限サンプルから未観測行動・将来報酬を完全復元するものではない。分布差が大きいと分散が爆発し、有限精度で破綻する。v73の極端なstale PPO軌跡の安全拒否・v79のnumeric guardで棄却したサンプルがある場合、無条件の期待値一致は主張できない。
5. v78 Critic Gossipは**パラメータ差分のみを共有**。他ピアの別Actorをtargetとして既に計算済みの重み差分を、受信側のActorへ数学的に再補正することは**できない**。この制約は今回も未解決。よって「全ての分散Critic更新が現Actorへ厳密補正された」とは主張しない。次段階では、旧方策log確率・軌跡・署名済み報酬を共有し、受信側が自身のtarget Actorで再重み付けするプロトコルが必要。または全ピアが同一の凍結target Actorに対してのみ更新し、同一スナップショットIDを検証する必要がある。
6. 今回の補正はCriticのMC価値損失についてであり、PPOのActor lossそのもの、原子的二重ジャーナル、HF実GPUの収束品質を保証しない。

## 注意

v79の `trajectory_is` は旧PPOのGAE value target / value clippingとは別の学習目標であり、既存のOptimizerチェックポイントを数学的に等価な学習軌道へ移行するものではない。新規ルーム推奨。

## 主なファイル

- `patchlib/offpolicy_critic_v79.py`: unclipped ISの期待値一致するCritic surrogate
- `patchlib/mixed_learning_v65.py`: rolloutと再利用Actor確率に基づくloss切替/ルームポリシー
- `patchlib/cli.py` と `patchlib/mix_gui_v66.py` と `web/desktop_webview.*`: CLI/GUI設定
- `tests/test_offpolicy_critic_v79.py`: 全列挙・PPO・HF Critic・GUI/CLIテスト
- `VALIDATION_V79.md`, `CHANGES_V79.patch`
