# Review follow-up: cache invariant and focused test matrix

Head: `51821c01ce95b4c0bfc087c0d1ae4885d61bbd0c`. Date: 2026-11-03.

- Documented that reference and suffix caches retain the quantizer output's dtype/device. The shortcut relies on that invariant; arbitrary mixed-dtype/device caches are not a new supported contract.
- Reduced the Cartesian test matrix from 120 to 30 by removing the BF16/noncontiguous axes. Retained reference lengths 0/1/72, history lengths 0/1/2/25/72 and batch sizes 1/4. Tests still compare exact convolution input, input immutability and the intentionally reduced pre-conv cat count.
- Added a comment beside the cat-count assertion explaining the regression it guards.
- Production executable AST is identical to the previously measured candidate; only comments changed. Historical A10 timing samples remain unchanged and are not a new full-model speedup claim. KV-cache deepcopy and the other concatenations remain.

## Results

**30 passed, 39 deselected** for the focused function; **118 passed** for both related modules. Zero failures/errors/skips. The latter includes the 30 additions. All 13 applicable pre-commit hooks exited 0 with no SKIP, and the tracked source stayed unchanged. Logs normalize only line endings/trailing whitespace; JUnit omits ephemeral host identity. Numeric results and assertions are retained.

The same local RTX4060 CUDA-enabled environment was reused: vLLM0.32.0, Torch2.13.0+cu130, CUDA13.0, cuDNN9.20, Transformers5.19.0, Python3.12.3. Exact inventory and warnings are retained. Shallow-checkout/editable version labels can lag the source commit; the runner checks HEAD and both source hashes. These remain CPU-marked repository tests with available cuDNN, not full GPU inference or CUDA Graph performance tests.

## Reproduce in the matching development environment

```bash
git checkout 51821c01ce95b4c0bfc087c0d1ae4885d61bbd0c
python -m pytest tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py \
  -k suffix_pre_conv_context_avoids_full_history_cat -m 'core_model and cpu' --run-level=core_model -q
python -m pytest tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py \
  tests/model_executor/models/qwen3_tts/test_qwen3_tts_code2wav.py \
  -m 'core_model and cpu' --run-level=core_model -q
```

Prerequisites: Linux, Python3.12, matching CUDA-enabled vLLM0.32/PyTorch dependencies and pytest plugins. No model weights or running server. The helper `run_cuda_review1_checks.py --source /path/to/checkout --result-dir /path/to/results` adds hash/environment checks and saves logs/JUnit automatically. Prior 120/208 results remain in the previous evidence directory, scoped to the prior head.
