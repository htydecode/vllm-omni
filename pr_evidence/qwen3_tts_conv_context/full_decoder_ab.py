"""Real-weight segmented decoder A/B, using the currently imported source.

Random fixed codec IDs are numerical fixtures, not natural speech or quality evidence.
The server and model source files are never modified by this program.
"""

import argparse
import ast
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import MethodType


MODULE = "vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
ORIGINAL = "conv_context = torch.cat([ref_hidden, old_quantized], dim=-1)[:, :, -_CONV_CONTEXT_FRAME:]"
OPTIMIZED = """if old_quantized.shape[-1] >= _CONV_CONTEXT_FRAME:
    conv_context = old_quantized[:, :, -_CONV_CONTEXT_FRAME:]
else:
    conv_context = torch.cat([ref_hidden, old_quantized], dim=-1)[:, :, -_CONV_CONTEXT_FRAME:]"""


def build_methods(text, namespace):
    tree = ast.parse(text)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3TTSTokenizerV2Decoder"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_decode_suffix"
    )
    before = ast.parse(ORIGINAL).body[0]
    after = ast.parse(OPTIMIZED).body[0]
    candidates = [
        i
        for i, n in enumerate(fn.body)
        if ast.dump(n) in {ast.dump(before), ast.dump(after)}
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Unsupported convolution-context implementation; do not force this A/B transform"
        )
    methods = {}
    for name, statement in [("baseline", before), ("candidate", after)]:
        variant = copy.deepcopy(fn)
        variant.body[candidates[0]] = copy.deepcopy(statement)
        variant.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[variant], type_ignores=[]))
        scope = dict(namespace)
        exec(compile(module, "<installed-source-" + name + ">", "exec"), scope)
        methods[name] = scope["_decode_suffix"]
    return methods


def load_weights(decoder, tokenizer_dir):
    from safetensors import safe_open

    expected = set(decoder.state_dict())
    loaded = set()
    files = sorted(tokenizer_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors checkpoint files in {tokenizer_dir}")
    for file in files:
        tensors = {}
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith("decoder."):
                    continue
                local_key = key[len("decoder.") :]
                if local_key not in expected or local_key in loaded:
                    raise RuntimeError(f"Unexpected or duplicate decoder weight: {key}")
                tensors[local_key] = f.get_tensor(key)
                loaded.add(local_key)
        if tensors:
            decoder.load_state_dict(tensors, strict=False)
    if loaded != expected:
        raise RuntimeError(
            f"Missing decoder weights, first 20: {sorted(expected - loaded)[:20]}"
        )
    print("Loaded decoder state tensors:", len(loaded), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    p.add_argument("--modes", nargs="+", choices=["eager", "graph"], default=["eager"])
    p.add_argument(
        "--reference-modes", nargs="+", choices=["xvec", "icl"], default=["xvec", "icl"]
    )
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        p.error("Invalid repeats/warmup")
    import torch

    if not torch.cuda.is_available():
        p.error("Use the A10 CUDA environment")
    device = torch.device(args.device)
    if device.type != "cuda":
        p.error("Expected a CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    module = importlib.import_module(MODULE)
    path = Path(inspect.getfile(module))
    text = path.read_text(encoding="utf-8")
    if "# TTS_CONTEXT_DIAGNOSTIC_V1" in text:
        raise RuntimeError("Restore temporary diagnostics before timing this benchmark")
    methods = build_methods(text, vars(module))
    config_path = args.tokenizer_dir / "config.json"
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_module = importlib.import_module(
        MODULE.replace("modeling_", "configuration_")
    )
    dtype = getattr(torch, args.dtype)
    decoders = {}
    checkpoint_manifest = {
        f.name: f.stat().st_size
        for f in sorted(args.tokenizer_dir.glob("*.safetensors"))
    }
    with torch.inference_mode():
        for name in ["baseline", "candidate"]:
            torch.manual_seed(42)
            config = config_module.Qwen3TTSTokenizerV2DecoderConfig(
                **raw_config["decoder_config"]
            )
            config._attn_implementation = "sdpa"
            decoder = module.Qwen3TTSTokenizerV2Decoder(config).eval()
            load_weights(decoder, args.tokenizer_dir)
            decoder = decoder.to(device=device, dtype=dtype)
            decoder._incremental_chunk_frames = 25
            if hasattr(decoder, "precompute_snake_caches"):
                decoder.precompute_snake_caches()
            decoder._decode_suffix = MethodType(methods[name], decoder)
            decoders[name] = decoder

        # One first frame, growing suffix, rolling window, and a partial final block.
        lengths = [1, 25, 25, 25, 25, 25, 7]
        cases = {}
        generator = torch.Generator(device="cpu").manual_seed(42)
        for reference_mode in args.reference_modes:
            prefix = (
                0 if reference_mode == "xvec" else min(72, int(config.sliding_window))
            )
            codes = torch.randint(
                0,
                config.codebook_size,
                (1, config.num_quantizers, prefix + sum(lengths)),
                generator=generator,
            )
            chunks = [codes[..., : prefix + lengths[0]].to(device)]
            offset = prefix + lengths[0]
            for length in lengths[1:]:
                chunks.append(codes[..., offset : offset + length].to(device))
                offset += length
            cases[reference_mode] = (prefix, chunks)

        def sequence(decoder, prefix, chunks, retain):
            cache = {"prefix_frames": prefix}
            outputs = []
            for chunk in chunks:
                output = decoder.chunked_decode(chunk, caches=cache)
                if retain:
                    outputs.append(output.clone())
            return outputs, cache

        def compare_caches(left, right):
            # Check all persistent tensor state used by subsequent suffix decoding.
            for key in [
                "ref_hidden",
                "ref_conv",
                "prefix_hidden",
                "suffix_quantized",
                "suffix_conv",
            ]:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            for key in ["suffix_frames", "decoder_prefix_frames"]:
                if left.get(key) != right.get(key):
                    raise AssertionError(f"Cache metadata mismatch: {key}")
            lcache, rcache = left["past_key_values"], right["past_key_values"]
            if len(lcache.layers) != len(rcache.layers):
                raise AssertionError("KV layer count mismatch")
            for llayer, rlayer in zip(lcache.layers, rcache.layers):
                for name in ["keys", "values"]:
                    a, b = getattr(llayer, name), getattr(rlayer, name)
                    if a is None or b is None:
                        if a is not b:
                            raise AssertionError(f"KV {name} initialization mismatch")
                    else:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

        rows = []
        for mode in [m for m in ["eager", "graph"] if m in args.modes]:
            if mode == "graph":
                for decoder in decoders.values():
                    decoder.enable_cudagraph(
                        capture_modes=tuple(args.reference_modes),
                        capture_batch_sizes=[1],
                        device=device,
                        codec_chunk_frames=25,
                        initial_codec_chunk_frames=1,
                        async_chunk=True,
                    )
                    wrapper = decoder._cudagraph_wrapper
                    wrapper._stats_enabled = True
                    wrapper._stats_log_every = 0
                    wrapper._stats_file = ""
                    wrapper._stats_replays.clear()
            for reference_mode, (prefix, chunks) in cases.items():
                # Interleave A/B after EACH chunk, including all persistent caches.
                caches = {name: {"prefix_frames": prefix} for name in decoders}
                max_abs = 0.0
                samples = []
                for chunk in chunks:
                    outputs = {
                        name: decoder.chunked_decode(chunk, caches=caches[name]).clone()
                        for name, decoder in decoders.items()
                    }
                    torch.testing.assert_close(
                        outputs["baseline"], outputs["candidate"], rtol=0, atol=0
                    )
                    compare_caches(caches["baseline"], caches["candidate"])
                    max_abs = max(
                        max_abs,
                        float(
                            (outputs["baseline"] - outputs["candidate"])
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    samples.append(int(outputs["baseline"].shape[-1]))
                graph_proof = {}
                if mode == "graph":
                    for name, decoder in decoders.items():
                        wrapper = decoder._cudagraph_wrapper
                        graph_proof[name] = [
                            {"key": list(k), "replays": n}
                            for k, n in wrapper._stats_replays.items()
                        ]
                        if not any(
                            k[0].startswith("suffix_") and n > 0
                            for k, n in wrapper._stats_replays.items()
                        ):
                            raise RuntimeError(
                                "Graph mode fell back without suffix replays; do not claim graph validation"
                            )
                        wrapper._stats_replays.clear()
                        wrapper._stats_enabled = False
                for decoder in decoders.values():
                    for _ in range(args.warmup):
                        sequence(decoder, prefix, chunks, False)
                timings = {"baseline": [], "candidate": []}
                for repeat in range(args.repeats):
                    order = (
                        ["baseline", "candidate"]
                        if repeat % 2 == 0
                        else ["candidate", "baseline"]
                    )
                    for name in order:
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        sequence(decoders[name], prefix, chunks, False)
                        torch.cuda.synchronize(device)
                        timings[name].append((time.perf_counter() - started) * 1000)
                # Turn counts back on only for the next correctness pass.
                if mode == "graph":
                    for decoder in decoders.values():
                        decoder._cudagraph_wrapper._stats_enabled = True
                medians = {k: statistics.median(v) for k, v in timings.items()}
                row = {
                    "mode": mode,
                    "reference_mode": reference_mode,
                    "chunk_lengths": lengths,
                    "prefix_frames": prefix,
                    "output_samples_by_chunk": samples,
                    "max_abs_error": max_abs,
                    "exact_waveform_and_cache_match": True,
                    "samples_ms_per_sequence": timings,
                    "median_ms": medians,
                    "graph_correctness_replays": graph_proof,
                }
                rows.append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "samples_ms_per_sequence"}
                    ),
                    flush=True,
                )
    report = {
        "scope": "real-weight segmented decoder, fixed random codec IDs, batch 1, not serving or audio quality",
        "source_file": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_files_bytes": checkpoint_manifest,
        "attention": "sdpa",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
