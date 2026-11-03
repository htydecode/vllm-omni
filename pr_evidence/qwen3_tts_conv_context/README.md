# Qwen3-TTS convolution-context evidence

This bundle supports a narrow optimization: avoid full-history concatenation when selecting two convolution-context frames. It is separate from the two-file PR diff.

## Provenance

- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, as identified by the contributor. Checkpoint revision was not recorded.
- Hardware: one NVIDIA A10; BF16; PyTorch `2.13.0+cu130`, CUDA `13.0`.
- Installed decoder checks: Python `3.12.3`, contributor-reported vLLM-Omni `0.30`; exact vLLM/Transformers versions were not captured.
- Installed model-source SHA-256: `cb692d0c4a9b0b6aac797a78fcd39be085b0cc5a127ac508843d6fd5daf3a9b3`.
- Microbenchmark source base: `630bd10d83b4c6942f11fdc5fb49394096daac10`; that model file is byte-identical to the PR base `7afd00e982dfa93c365eee5475a5707904b181e6`.
- `baseline.py` SHA-256: `5be299858d3d242eb060f05dd643bf81582dee47247403afd35699fcbd12e990`.
- `candidate.py` SHA-256: `d0b46501c11ff0239970198c71f01e5fcf295d8a6fe0939e1a813b005a70886f`.

The contributor supplied JSON/JSONL data through a plain-text export. Public copies retain numeric values, samples and source hashes; server absolute paths are replaced with `server-file:<basename>`. `path-counts.json` aggregates the supplied request windows without publishing PIDs or wall-clock timestamps. Raw local diagnostics, audio and recovery manifests are excluded.

## Reproduce the local context benchmark

Run from this directory in a CUDA PyTorch environment on an idle GPU. No vLLM import or checkpoint is required. The quantizer is identity; only the context extraction and next concatenation execute.

```bash
python bench_tts_conv_context.py \
  --baseline baseline.py --candidate candidate.py \
  --channels 512 --device cuda:0 --dtype bfloat16 \
  --warmup 50 --iterations 100 --repeats 5 \
  --output results/new-run.json
```

Repeat in five independent processes, then summarize their reports with `summarize_repeats.py`. Use a fresh directory containing only those raw reports. The five supplied reports and their summary are in `results/local-repeats/`.

## Reproduce standalone tensor checks

`full_decoder_ab.py` provides a source-only AST transform imported by the tensor tests; importing the helper does not import the model. Use a restored, compatible installed model source and CPU PyTorch/pytest:

```bash
TTS_CONTEXT_SOURCE=/path/to/installed/modeling_qwen3_tts_tokenizer_v2.py \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
python -m pytest -c /dev/null --confcutdir "$PWD" \
  test_context_tensor.py -q
```

The supplied `results/tensor-tests-w5eww4p2/summary.json` records 120 tests without failures/errors/skips. It is a contributor report of this standalone harness, not a passing result for the PR's repository integration module. Pytest text and JUnit reports were not included in the supplied export.

## Reproduce complete decoder checks

This uses the currently imported server implementation and normal `decoder.*` safetensors weights. It constructs the two method variants in memory, without modifying installed source. Requires a compatible vLLM-Omni development/serving environment and the local checkpoint. The published reports used SDPA, batch 1 and fixed random codec IDs; they do not assess natural-speech quality.

```bash
python full_decoder_ab.py \
  --tokenizer-dir /path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer \
  --device cuda:0 --dtype bfloat16 --modes eager graph \
  --repeats 10 --warmup 2 --output results/new-decoder-ab.json
```

The test compares per-chunk waveforms, persistent tensor/KV state, growing/rolling windows and the partial tail, and verifies actual suffix graph replays. It does not compare legacy and new streaming backends. See `results/decoder-eager.json` and `results/decoder-eager-graph.json` for every supplied timing sample, including regressions. Complete-decoder results are mixed; no consistent complete-decoder or E2E speedup is claimed.

`trace_suffix.py` is a temporary diagnostic helper for isolated service requests. It is not production instrumentation and must be restored before performance timing. Supplied path counts are in `results/path-counts.json`.

No VRAM, E2E, TTFA or perceptual-quality improvement is established. The small eager fallback slowdown and graph/ICL complete-decoder regression are retained rather than omitted.
