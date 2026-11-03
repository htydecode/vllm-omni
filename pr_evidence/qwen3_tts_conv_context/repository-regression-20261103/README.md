# Matching-head repository regression

Validation date: 2026-11-03. Head: `bff6f65ac32fba6739b8327d7caa41996bf7d1b6`.
These results supplement the original A10 component/full-decoder reports; they do not replace or reattribute those measurements.

## Results and scope

- Targeted added test: **120 passed, 39 deselected, 0 failures/errors/skips**, 0.64 s.
- Both incremental-decode and Code2Wav files: **208 passed, 0 failures/errors/skips**, 3.46 s. This includes the 120 cases; do not add the counts.
- The two `test_decode_autotune_restores_process_flags` variants both passed. An earlier CPU-only PyTorch run had 206 passes/2 failures, also reproduced on the clean parent: its unavailable cuDNN leaves a monkeypatched benchmark_limit at 5 while the tests expect 10. The passing run uses available cuDNN and keeps the assertions unchanged.
- All 13 hooks applicable to the two changed Python paths, including the suggestion hook, exited 0. They ran individually without SKIP in the local development container. Nonmatching Go/YAML/Markdown/shell hooks were not installed; full hook initialization had previously hit a Go-bootstrap TLS error. Source stayed unchanged.

The pytest cases retain `core_model and cpu` marks and their normal tensor allocation. Running them with CUDA-enabled PyTorch qualifies cuDNN flag handling, not 208 real GPU inference/graph cases. No model weights were loaded. This is not a new latency, throughput, memory or quality benchmark.

## Environment and provenance

Windows Docker Linux/WSL2, Ubuntu 24.04.3, glibc 2.39, Python 3.12.3. RTX 4060 Laptop 8 GB, driver 591.74. vLLM 0.32.0, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20, Transformers 5.19.0, pytest 9.1.1. `pip check` passed; the exact inventory is in results/pip-freeze.log, which includes local artifact locations and is not a portable requirements lock.

The cached `vllm/vllm-omni:v0.30.0` image supplied only the initial OS/runtime. The isolated container replaced vLLM with 0.32.0 and Omni with the fixed PR source. Its global UV_OVERRIDE pinned NCCL 2.30.7; only the final NCCL install used `env -u UV_OVERRIDE`, aligning to Torch's required 2.29.7. The image configuration and PR code were not changed.

The shallow checkout reports Omni as `0.1.dev2+gbff6f65ac`; source commit and both file hashes were checked explicitly. The logs retain the resulting version-label warning and 14 Torch JIT deprecation warnings. The CUDA wheel is verified against the official release SHA-256 before installation.

## Reproduce

On a Linux Docker host with an NVIDIA GPU and CUDA13-compatible driver, run from this directory:

```bash
git init source
git -C source fetch --depth=2 https://github.com/vllm-project/vllm-omni.git bff6f65ac32fba6739b8327d7caa41996bf7d1b6
git -C source checkout --detach FETCH_HEAD
mkdir -p out
docker run --gpus all \
  --mount type=bind,source="$PWD",target=/kit,readonly \
  --mount type=bind,source="$PWD/source/.git",target=/source-cache,readonly \
  --mount type=bind,source="$PWD/out",target=/out \
  --entrypoint /bin/bash vllm/vllm-omni:v0.30.0 /kit/run_cuda_regression.sh
```

The launcher installs matching dependencies and runs both pytest commands shown in results/summary.json. It downloads no model checkpoint. A preconfigured matching development checkout can use those commands directly. Evidence XML replaces only the ephemeral container hostname; all cases, counts, timings and warning text are retained. sha256.json covers the published files.
