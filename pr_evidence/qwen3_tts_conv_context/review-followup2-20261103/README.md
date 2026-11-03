# Second review follow-up: strided BF16 and executable cache invariants

Head: `7cacc5e46d8c84651f9e15eaceb7b53a4a80892c`. Validation date: 2026-11-03.

The 30 FP32/contiguous combinations remain. Exactly one case adds batch=1, ref_frames=1, old_frames=25 with strided BF16 reference/history/new tensors. It asserts that the tensors are genuinely noncontiguous, and checks exact convolution input, dtype/device preservation, input immutability and the intentionally reduced cat count. This is **31 cases**, not a reinstated cross product.

The existing real incremental audio/full-decode test now asserts reference/suffix cache dtype and device equality after every chunk, for xvec and ICL. Those checks run on the actual cache initialization/update path rather than only checking synthetic test inputs. The production decoder and hot-path behavior are unchanged; no per-chunk production assertions were added.

## Results

- Targeted suffix test: **31 passed, 39 deselected**, no failures/errors/skips.
- Incremental-decode and Code2Wav modules together: **119 passed**, no failures/errors/skips; includes those 31.
- All 13 applicable changed-file pre-commit hooks passed with no SKIP; tracked diff is empty.
- Two separate mutation diagnostics replace only the in-process forward method to corrupt the returned reference cache. Changing its dtype makes both xvec/ICL cases fail at the new dtype assertion. Changing its device to meta makes both fail at the new device assertion. Each diagnostic expects pytest exit=1 and checks the exact assertion text; each reports mutation_detected=true. These **intentional negative-control failures are not ordinary regression failures**. No production file is modified, and the method is restored after each diagnostic.

Same local CUDA-enabled environment: RTX4060 Laptop, Python3.12.3, vLLM0.32.0, Torch2.13.0+cu130, CUDA13.0, cuDNN9.20, Transformers5.19.0. The tests retain CPU markers; this is not full-model GPU inference/graph/performance validation. Editable/shallow-build labels may lag the source SHA; actual HEAD and file hashes are explicitly checked. Original warning text is retained.

## Reproduce

Prerequisites: a Linux development checkout at the head above, compatible vLLM0.32/CUDA PyTorch and the repository pytest plugins. No checkpoint or service is needed.

```bash
python -m pytest tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py \
  -k suffix_pre_conv_context_avoids_full_history_cat -m 'core_model and cpu' --run-level=core_model -q
python -m pytest tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py \
  tests/model_executor/models/qwen3_tts/test_qwen3_tts_code2wav.py \
  -m 'core_model and cpu' --run-level=core_model -q
python run_cache_invariant_mutations.py --source /path/to/checkout --result-dir /path/to/results --mutation dtype
python run_cache_invariant_mutations.py --source /path/to/checkout --result-dir /path/to/results --mutation device
```

The helper run_cuda_review2_checks.py automates environment/source verification and normal pytest logs/JUnit. Public logs normalize only LF/trailing whitespace; JUnit replaces ephemeral hostnames. Numeric outcomes and assertions are retained. Historical A10 measurements and the preceding validation runs retain their original source/environment scopes. No new performance measurement is claimed because this follow-up changes tests only. sha256.json covers this supplement.
