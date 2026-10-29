# v63 validation

- Full `pytest -q --disable-warnings` run after principal Python compatibility changes: **768 passed, 3 skipped, 0 failed** in 92.73 s.
- Following the final NEW MODEL UI/default-initialization guard, 89 focused tests from model loading, v60/v61/v62 offload, LoRA-Muon, CLI, TinyStories GUI, WebView, and elastic integrations: **89 passed, 0 failed**.
- Added 17 `test_hf_compat_v63.py` tests: BF16 snapshot load + rebuilt RoPE and tied weights; PyTorch autograd + optimizer update; strict dtype failure; unsupported nonpersistent buffer fail-closed; preserving EOS targets; compatible Qwen2/Llama config checks; explicit DeepSeek native/quantization refusal; GUI streaming path and initialization; Muon generalized projection policy; offline diagnostic.
- Static Python `compileall` and `node --check web/desktop_webview.js` passed.
- **Not run:** real Transformers `AutoModelForCausalLM` loading, gated Hugging Face weights, DeepSeek FP8/MoE kernels, 8B/300B training, actual multi-device WAN tests. This code environment has no transformers dependency installed.
- Exact full log before last UI adjustment: `VALIDATION_V63_FULL_PYTEST.log`.
