# Room-ID-only Elastic P2P training

This mode replaces user-managed roster files, `--link`, private/public key provisioning, and explicit `--peer-id` with a single shared `--room-id`. It uses the existing Mummy signaling server and Python aiortc DataChannel transport. The Python network process **never executes remote model factory code**.

## Quick start

Install the project's `requirements-patches.txt` and install/build the Nim signaling server as described in `README.md`.

Generate a cryptographically random invite string **on one machine**:

```bash
python -m patchlib.cli room-create
```

Start the Mummy server:

```bash
nim c -d:release --threads:on --mm:orc -r src/main.nim
```

The following example assumes all participants already have the **same immutable base checkpoint** (`genesis.fstate`, with identical bytes), the same model/revision/configuration, and their own data (`train.txt`). If you have a different pretrained base, distribute it using the existing `shard-export` / `shard-exchange` / `shard-assemble` workflow before training. Initial base distribution has **not** been automated by this change.

**Alice**:

```bash
python -m patchlib.cli train-live \
  --room-id 'PASTE_THE_RANDOM_ROOM_ID' \
  --model sshleifer/tiny-gpt2 --revision PINNED_REVISION \
  --state-base genesis.fstate --text-file train-alice.txt \
  --kind loha --rank 32 --local-steps 16 \
  --journal alice.sqlite3 --rounds 0
```

**Bob** (on another computer; same invite/model/base; different local journal):

```bash
python -m patchlib.cli train-live \
  --room-id 'PASTE_THE_SAME_ROOM_ID' \
  --ws wss://YOUR_DOMAIN/ws \
  --model sshleifer/tiny-gpt2 --revision PINNED_REVISION \
  --state-base genesis.fstate --text-file train-bob.txt \
  --kind loha --rank 32 --local-steps 16 \
  --journal bob.sqlite3 --rounds 0
```

Set `--ws` explicitly for all remote participants; `ws://127.0.0.1:8080/ws` is only for localhost. The example above uses `--ws` on Bob; if Alice also connects remotely, she needs it too.

There is no `--peer-id`, `--signing-key`, `--self-public-key`, `--trusted-peer-key`, `--link`, or `--roster-file` in this mode. The application creates a **private** Ed25519 identity at `<journal>.identity.pem` (file permissions 0600). The public fingerprint is derived from that key and is discovered from the room lobby. Store journal + identity together; a new journal on a different device creates a new identity. Do **not** publish the `.identity.pem` file.

## Implementation

- Server accepts `{"type":"group-join","room":"...","peer":"p<fingerprint>","token":""}` on existing `/ws`, returns `group-members` and emits `group-peer-joined` / `group-peer-left` notifications.
- Server stores only in-memory connection/membership metadata, not model updates, checkpoints, or training data. Maximum **32 participants per group**, **1024 groups**, and **4096 concurrent sockets**.
- A node creates up to four symmetric ring-neighbor WebRTC connections; 32 participants require ~64 peer-to-peer links rather than 496. Peers exchange their missing signed update IDs and gossip verified immutable updates over SCTP DataChannels. Some data may traverse coturn if peer-to-peer NAT traversal fails.
- Each signed update contains its signing public key; `peer-id = hash(public key)` is verified before applying it. This permits a relay to send updates from an author who has since gone permanently offline, without a separate trust registry.
- `room-id` and SHA-256 of the **base checkpoint bytes** are included in the epoch/genesis fingerprint so same model/different checkpoint or room is not silently mixed.
- Disconnected peers are excluded from the *room-presence-based live divisor* (including members not directly connected as ring neighbors). Membership is not cryptographically authenticated by the signaling server and a malicious participant can inflate the count. Reconnect and late join can reconcile journal events subject to journal storage limits. This is an **asynchronous additive FedAvg-like approximation**, not mathematically strict consensus FedAvg.
- The server does not learn gradients/LoHA factors directly; transport remains DTLS/SCTP. It still sees membership and exchanges SDP/ICE, and **TURN relays can carry the full encrypted data volume** if used.

## Security and operational caveats

- **A room ID is a bearer secret**, not authorization tied to a person. Someone who knows it can join and attempt poisoning. Generate room IDs with `room-create`; do not use predictable strings. Passing room IDs as CLI args also exposes them in shell histories and potentially process listings.
- Signatures verify *which key created an update*, not whether that participant is honest or allowed. This is not Sybil-resistant. Dedicated admission/reputation and resource quotas are needed before allowing unknown users into a high-value training group.
- On public deployments use WSS through a reverse proxy with TLS, per-IP connection/rate limits, and coturn quotas. Mummy's in-memory room caps are not sufficient DDoS protection. For highly sensitive membership, add authenticated invitations rather than room-secret-only admission.
- Ensure the same immutable genesis checkpoint **on disk**; the base must be available on startup and stable across restarts. The existing journal still has a hard size cap and does not prune old events silently. Checkpoint epoch advancement and automatic bootstrap from unknown peers remain separate tasks.
- For best results use a pinned Hugging Face commit, not mutable `main`, and ensure the same tokenizer/model mapping. This change does not solve adversarial or anomalous ML update quality by itself.

## Testing

```bash
PYTHONPATH=. python -m pytest -q
```

The Python test suite covers generated keys, key-binding, signed third-party replay, room membership/rejoin, offline learning, and a mocked WebRTC DataChannel handshake. Nim compilation, coturn, and a multi-host WebRTC run must still be tested in your environment.
