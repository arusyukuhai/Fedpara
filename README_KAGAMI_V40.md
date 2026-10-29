# Kagami v40 — 世代チェックポイントと更新ログの安全な退役

## 目的と現在の対応範囲

v39までのElastic/Gossip v1では、更新イベントの全履歴が同期と再起動の基準となっています。**実行中の同一ルームで `.fplog` を切り詰めると、離席ピアの再接続、署名検証、復元が破壊されます。** このため、v40は次の**停止して切り替える世代管理**を実装します。

1. トレーナーと中継を停止し、GUIの保存済み実効重み（`*.shards.json`）を完成させる。
2. そのスナップショットが、旧ジャーナルの**最終コミット境界**に一致することを確認する。所有者のEd25519鍵でチェックポイント記述子を署名する。
3. 署名、イベント一覧ハッシュ、実効重みのシャードを検証し、新しい世代の基準モデルを作る。新しいルームIDを割り当て、旧ルームと通信させない。
4. **全参加端末**が同一の世代フォルダー（内容ハッシュ）を取得し、新しいルームへ移る。GUIのモデル一覧に新世代をインポートできる。
5. 新世代で正常に更新できたことを検証した後、運営者が明示的に承認した場合に限り、旧ログとそのSQLiteイベント索引を退役・削除できる。

現在の **v1 Gossip のまま、学習を止めずに各ピアが独立して古い更新を削除する自動ガベージコレクションは未実装**です。全オンライン・オフライン参加者が同一世代を採用することをプログラムで合意する分散合意機構もまだありません。従って自動でログ削除はしません。旧世代のログ削除はオペレーターの移行確認が必要です。

## 使い方（一般的なGUIで学習したモデル）

### 1. 学習を安全に終了し、最新の保存完了ログを確認

学習を一時停止するだけではなく、**中継も含めた旧世代のプロセスをすべて終了**してください。全ピアについて同じ世代の更新を取り込んだかを確認してください。v1では一部ピアの未同期イベントを新しい基準に取り込んだことを自動保証できません。

v40で学習を正常終了すると、GUIで保存された `effective-...shards.json` に対応する署名付き `.checkpoint.json` が自動生成されます（署名済みルームの場合）。この場合 `seal` は省略できます。GUIで保存された `effective-...shards.json` と対応するシャード格納ディレクトリ、およびそのピアの `events.sqlite3`（ジャーナル）が必要です。**旧 `.fedpara-gui` と `.identity.pem` を削除しないでください。**

### 2. 保存した重みに署名付きの世代境界を付与

```bash
python -m patchlib.generation_rollover seal \
  --journal /path/old-events.sqlite3 \
  --checkpoint /path/effective-xxxx.shards.json \
  --store /path/old-model-objects
```

スナップショットの `seq` とジャーナルのコミットイベント数が一致し、チェックポイントが署名済みでシャードを検証できる場合にのみ成功します。任意の保存ファイルを事後的に『正しい学習成果』と暗号学的に証明するものではありません。元トレーナーの正しい保存処理を前提とします。

既に署名付き `elastic-...fmanifest` と `.checkpoint.json` がある場合、`seal` は不要です。ただし最後の更新まで含むことが必要です。

### 3. 新しい世代を準備

```bash
python -m patchlib.generation_rollover prepare \
  --journal /path/old-events.sqlite3 \
  --checkpoint /path/effective-xxxx.shards.json \
  --store /path/old-model-objects \
  --room-manifest /path/old-room/manifest.json \
  --old-room-id OLD_ROOM_ID \
  --destination /path/kagami-generation-2
```

この操作は**旧ログを一切消しません**。新しい世代フォルダーには `base.fmanifest`、`store/`、`room-manifest.json`、`generation.json` が作られます。シャードは同一ファイルシステム上ではハードリンクされ、不要な二重保存を避けます。別ファイルシステムでは検証後にコピーします。旧イベントは新世代に移しません。

### 4. 全ピアで同じ世代ファイルを使って開始

世代フォルダー全体を各端末に配布し、各端末で `generation.json` の `base_semantic_sha256` と `base_sha256`、全シャードを確認してから以下を実行します。

```bash
python -m patchlib.generation_rollover import-gui \
  --generation /path/kagami-generation-2 \
  --workdir /path/your/KAGAMI_DATA_DIR
python kagami.py
```

GUIのMODELSで新しいルームIDのモデルを選んで開始すると、`--sharded-base-manifest` と `--sharded-base-store` を指定して新しい初期重みから学習します。各端末には新しい専用ジャーナルが作られ、旧プロファイルの署名鍵・更新索引と混ざりません。

**モデルをロードする際、モデル構造は各端末のTransformers環境で復元されます。** シャードを直接ロードするため、モデル全体を収容できるメモリまたは別途対応するオフロード構成が必要です。今回のコードだけで任意の300Bモデルが省メモリ起動できるとは確認していません。

### 5. 旧ログを退役（削除を伴う、不可逆）

**全ピアの世代移行、旧ルーム停止、新世代からの再起動・再接続が成功したことを確認した後**、明示的に次を実行してください。

```bash
python -m patchlib.generation_rollover retire \
  --journal /path/old-events.sqlite3 \
  --checkpoint /path/effective-xxxx.shards.json \
  --store /path/old-model-objects \
  --generation /path/kagami-generation-2 \
  --new-journal /path/new-generation-events.sqlite3 \
  --confirm-new-room-id NEW_ROOM_ID
```

この操作は新世代のシャードの再検証、元ジャーナルの不変性、新しいジャーナルに署名付き更新が存在することを確認したうえで、**旧 `.fplog` と旧イベントSQLiteデータベースを削除**します。旧モデルを元の旧ルームで再起動できなくなります。**通信網のすべての端末が新世代へ移行したという保証は、コードからは得られません。**

退役後も、新しい `base.fmanifest` と `store/` は継続学習に必須のため消してはいけません。世代ファイルを消すと新しいモデルを再起動できません。

## 安全確認と互換性

- 停止していないジャーナルの変更・移行はOSの排他ロックで拒否します。
- イベント一覧の完全性・署名・最新seq・シャードSHA-256を確認できない場合は移行しません。
- 新旧ルームIDは必ず別です。異なる世代を同じGenesisとして誤接続しません。
- 旧ジャーナルを含むv39のモデルはそのまま残り、既存GUIの起動処理は変更しない設計です。
- シャード初期重みのCLI指定をElasticモードでも許可し、初回起動・再起動で同じパスを固定します。
- 一度に配布するチェックポイントが非常に巨大な場合の転送はこのオフラインワークフローの外側です。

## 次に必要なオンライン自動化

連続稼働しながら **定期チェックポイント → 共有世代の合意 → 遅れて復帰した端末へ証明付き基準を配布 → 世代確定後の自動GC** を行うためには、全ピアが検証できる世代コミット証明、離脱ピアの古い更新の扱い、実行中更新のフリーズと再計算、モデル同一性の照合が必要です。既存の `epoch_consensus.py` はこの本番の切り替えにはまだ連動していません。ネットワークを分裂させるため、これらを未検証のまま自動有効化していません。
