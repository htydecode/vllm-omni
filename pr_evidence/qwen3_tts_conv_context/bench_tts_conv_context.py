"""Local A/B probe: real source prefixes through new_conv_input, not full TTS.

Only requires PyTorch. Does not import vLLM or download model weights.
"""
import argparse
import ast
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


def extract_prefix(path, torch):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3TTSTokenizerV2Decoder')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_suffix')
    stop = next(i for i, n in enumerate(method.body)
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'new_conv' for t in n.targets))
    method.body = method.body[:stop] + [ast.Return(value=ast.Name(id='new_conv_input', ctx=ast.Load()))]
    method.decorator_list = []
    method.returns = None
    for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs:
        arg.annotation = None
    prefix = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    constants = [n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_CONV_CONTEXT_FRAME' for t in n.targets)]
    if len(constants) != 1:
        raise ValueError('Expected one context-size constant')
    scope = {'torch': torch, '_CONV_CONTEXT_FRAME': ast.literal_eval(constants[0])}
    exec(compile(prefix, str(path), 'exec'), scope)
    return scope['_decode_suffix']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--channels', type=int, required=True, help='Use decoder_config.codebook_dim from the tokenizer config')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.channels, args.iterations, args.repeats) <= 0 or args.warmup < 0:
        p.error('Invalid measurement parameters')
    import torch

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is not available in this Python environment')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(42)
    methods = {name: extract_prefix(path, torch)
               for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    decoder = SimpleNamespace(config=SimpleNamespace(sliding_window=72),
                              quantizer=SimpleNamespace(decode=lambda codes: codes))
    rows = []
    cases = [(1, 0, 1), (1, 0, 2), (1, 0, 72), (1, 72, 72), (4, 0, 72), (4, 72, 72)]
    with torch.inference_mode():
        for batch, ref_len, old_len in cases:
            ref = torch.randn(batch, args.channels, ref_len, device=device, dtype=dtype)
            old = torch.randn(batch, args.channels, old_len, device=device, dtype=dtype)
            new = torch.randn(batch, args.channels, 25, device=device, dtype=dtype)
            def bind(method):
                return lambda: method(decoder, new, old, None, {'ref_hidden': ref}, ref_len, 25, False)
            eager = {name: bind(method) for name, method in methods.items()}
            torch.testing.assert_close(eager['baseline'](), eager['candidate'](), rtol=0, atol=0)
            modes = ['eager', 'graph'] if device.type == 'cuda' else ['eager']
            for mode in modes:
                runners = {}
                graphs = {}
                for name, fn in eager.items():
                    for _ in range(args.warmup):
                        fn()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    if mode == 'graph':
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = fn()
                        graphs[name] = (graph, output)
                        runners[name] = graph.replay
                    else:
                        runners[name] = fn
                if mode == 'graph':
                    for _ in range(3):
                        new.add_(0.5)
                        old.add_(-0.25)
                        ref.add_(1)
                        for graph, _ in graphs.values():
                            graph.replay()
                        torch.cuda.synchronize(device)
                        torch.testing.assert_close(graphs['baseline'][1], graphs['candidate'][1], rtol=0, atol=0)
                samples = {'baseline': [], 'candidate': []}
                for repeat in range(args.repeats):
                    order = ['baseline', 'candidate'] if repeat % 2 == 0 else ['candidate', 'baseline']
                    for name in order:
                        if device.type == 'cuda':
                            torch.cuda.synchronize(device)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                        started = time.perf_counter()
                        for _ in range(args.iterations):
                            runners[name]()
                        if device.type == 'cuda':
                            end.record()
                            end.synchronize()
                        wall_us = (time.perf_counter() - started) * 1e6 / args.iterations
                        samples[name].append({'wall_us_per_call': wall_us,
                                              'cuda_event_us_per_call': start.elapsed_time(end) * 1000 / args.iterations
                                              if device.type == 'cuda' else None})
                row = {'batch': batch, 'ref_frames': ref_len, 'old_frames': old_len,
                       'new_frames': 25, 'mode': mode, 'samples': samples,
                       'median_wall_us': {k: statistics.median(s['wall_us_per_call'] for s in v)
                                          for k, v in samples.items()}}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
    report = {'torch': torch.__version__, 'device': str(device), 'dtype': args.dtype,
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'channels': args.channels, 'iterations': args.iterations, 'repeats': args.repeats,
              'scope': 'context extraction plus next concatenation; identity quantizer; not full decoder or TTS',
              'baseline_file': str(args.baseline), 'candidate_file': str(args.candidate), 'results': rows}
    report['baseline_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    report['candidate_sha256'] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
