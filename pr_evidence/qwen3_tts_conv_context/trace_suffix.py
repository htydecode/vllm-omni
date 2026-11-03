"""Temporary Qwen3-TTS diagnostics. Never include the instrumentation in a PR."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


MODEL_REL = Path(
    "model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
)
GRAPH_REL = Path("model_executor/models/qwen3_tts/segmented_graph_wrapper.py")
MARKER = "# TTS_CONTEXT_DIAGNOSTIC_V1"
HELPER = """
# TTS_CONTEXT_DIAGNOSTIC_V1
def _tts_context_diagnostic(event, **fields):
    import json as _json
    import os as _os
    import time as _time
    _path = _os.environ.get("TTS_CONTEXT_TRACE_FILE", "")
    if not _path or not _os.path.isfile(_path + ".enabled"):
        return
    # Python capture executions are not inference replays.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    _line = _json.dumps(dict(event=event, pid=_os.getpid(), time_ns=_time.time_ns(), **fields)) + "\\n"
    _fd = _os.open(_path, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
    try:
        _data = _line.encode("utf-8")
        if _os.write(_fd, _data) != len(_data):
            raise RuntimeError("Short diagnostic log write")
    finally:
        _os.close(_fd)
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package_root(explicit=None):
    if explicit:
        root = Path(explicit).resolve()
    else:
        import vllm_omni

        roots = [
            Path(p).resolve()
            for p in vllm_omni.__path__
            if (Path(p) / MODEL_REL).is_file()
        ]
        if len(roots) != 1:
            raise RuntimeError(
                f"Cannot choose installed vllm_omni path: {roots}; use --package-root"
            )
        root = roots[0]
    if not (root / MODEL_REL).is_file():
        raise RuntimeError(f"Expected package file not found: {root / MODEL_REL}")
    return root


def method(tree, owner, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def transform_model(text):
    if MARKER in text:
        raise RuntimeError("Already instrumented")
    fn = method(ast.parse(text), "Qwen3TTSTokenizerV2Decoder", "_decode_suffix")
    names = {a.arg for a in fn.args.args}
    if not {"old_quantized", "new_codes", "new_frames"}.issubset(names):
        raise RuntimeError(
            "Unsupported _decode_suffix signature; do not force this patch"
        )
    index = fn.body[0].lineno - 1
    if (
        isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        index = fn.body[0].end_lineno
    line = (
        '        _tts_context_diagnostic("suffix_python", old_frames=int(old_quantized.shape[-1]), '
        "batch=int(old_quantized.shape[0]), new_frames=int(new_frames), "
        "context_frames=int(_CONV_CONTEXT_FRAME))\n"
    )
    lines = text.splitlines(keepends=True)
    lines.insert(index, line)
    result = "".join(lines).rstrip() + "\n" + HELPER
    ast.parse(result)
    return result


def transform_graph(text):
    if MARKER in text:
        raise RuntimeError("Already instrumented")
    fn = method(ast.parse(text), "CUDAGraphDecoderWrapper", "_decode_suffix_batch")
    nodes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value) == "state['graph'].replay()"
    ]
    if len(nodes) != 1 or "static_quantized" not in {
        n.id for n in ast.walk(fn) if isinstance(n, ast.Name)
    }:
        raise RuntimeError(
            "Unsupported suffix graph implementation; do not force this patch"
        )
    line = (
        '        _tts_context_diagnostic("suffix_graph_replay", old_frames=int(static_quantized.shape[-1]), '
        "batch=int(static_quantized.shape[0]), requests=len(codes_list), "
        "target_frames=int(target_frames), mode=mode, context_frames=2)\n"
    )
    # The context-size check is version-specific and verified by instrument().
    lines = text.splitlines(keepends=True)
    lines.insert(nodes[0].end_lineno, line)
    result = "".join(lines).rstrip() + "\n" + HELPER
    ast.parse(result)
    return result


def inspect_install(args):
    root = package_root(args.package_root)
    for rel in [MODEL_REL, GRAPH_REL]:
        p = root / rel
        print(p)
        if p.is_file():
            print("  sha256:", digest(p.read_bytes()))
    text = (root / MODEL_REL).read_text(encoding="utf-8")
    print("Diagnostic patch present:", MARKER in text)
    print("Use the same Python executable and source environment as the service.")


def instrument(args):
    root = package_root(args.package_root)
    tree = ast.parse((root / MODEL_REL).read_text(encoding="utf-8"))
    values = [
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_CONV_CONTEXT_FRAME" for t in n.targets
        )
    ]
    if len(values) != 1 or ast.literal_eval(values[0]) != 2:
        raise RuntimeError("This diagnostic is only validated for a two-frame context")
    changes = []
    for rel, transform in [(MODEL_REL, transform_model), (GRAPH_REL, transform_graph)]:
        path = root / rel
        if not path.is_file():
            raise RuntimeError(
                f"Expected graph wrapper is missing: {path}; no files changed"
            )
        before = path.read_bytes()
        after = transform(before.decode("utf-8")).encode("utf-8")
        backup = path.with_name(path.name + ".tts-context-original")
        if backup.exists():
            raise RuntimeError(f"Backup already exists: {backup}; restore first")
        changes.append((path, backup, before, after))
    manifest = args.manifest.resolve()
    if manifest.exists():
        raise RuntimeError(f"Manifest already exists: {manifest}; choose a new path")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "path": str(p),
            "backup": str(b),
            "original_sha256": digest(old),
            "instrumented_sha256": digest(new),
        }
        for p, b, old, new in changes
    ]
    # Write the recovery manifest before changing source files.
    with manifest.open("x", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    for path, backup, before, after in changes:
        with backup.open("xb") as f:
            f.write(before)
        path.write_bytes(after)
        print("Instrumented:", path)
    print("Recovery manifest:", manifest)
    print(
        "Restart the backend with TTS_CONTEXT_TRACE_FILE set before running requests."
    )


def restore(args):
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    pending = []
    for row in rows:
        path, backup = Path(row["path"]), Path(row["backup"])
        current = digest(path.read_bytes())
        if current == row["original_sha256"]:
            print("Already original:", path)
            continue
        if current != row["instrumented_sha256"]:
            raise RuntimeError(
                f"Source changed after instrumentation; refusing to overwrite: {path}"
            )
        data = backup.read_bytes()
        if digest(data) != row["original_sha256"]:
            raise RuntimeError(f"Backup checksum mismatch: {backup}")
        pending.append((path, data))
    for path, data in pending:
        path.write_bytes(data)
        print("Restored:", path)
    print(
        "Backups retained. Restart the backend; unset TTS_CONTEXT_TRACE_FILE for formal timing."
    )


def append_event(path, **data):
    data["time_ns"] = time.time_ns()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def send_request(args, text, label, gate):
    body = {
        "model": args.model,
        "input": text,
        "voice": args.voice,
        "task_type": "CustomVoice",
        "language": "Chinese",
        "response_format": "wav",
        "stream": False,
    }
    if args.payload:
        body = json.loads(args.payload.read_text(encoding="utf-8"))
        body["input"] = text
        body["model"] = args.model
    print("Request:", label, flush=True)
    started = time.perf_counter()
    if gate:
        append_event(args.trace, event="client_begin", label=label)
        Path(str(args.trace) + ".enabled").write_text(label, encoding="utf-8")
    try:
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/audio/speech",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            data = response.read()
            content_type = response.headers.get("Content-Type", "")
        if not data or "json" in content_type or content_type.startswith("text/"):
            raise RuntimeError(
                f"Expected raw audio, got {content_type}: {data[:300]!r}"
            )
        path = args.out / (label + (".wav" if data[:4] == b"RIFF" else ".audio"))
        path.write_bytes(data)
        print("Saved:", path, "bytes=", len(data))
    except urllib.error.HTTPError as exc:
        print("HTTP error response:", exc.read().decode("utf-8", errors="replace"))
        raise
    finally:
        if gate:
            Path(str(args.trace) + ".enabled").unlink(missing_ok=True)
            append_event(
                args.trace,
                event="client_end",
                label=label,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )


def requests(args):
    if args.trace.exists() or Path(str(args.trace) + ".enabled").exists():
        raise RuntimeError(
            "Use a new trace path and restart the backend with that same path"
        )
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=False)
    short = "你好，欢迎使用语音助手。"
    long = (
        "今天我们进行一次较长的语音合成测试。请自然、清晰地朗读下面的内容。"
        "北京是一座历史悠久的城市，既有故宫和长城这样的文化遗产，也有现代化的交通和生活设施。"
        "学习推理优化时，我们先建立可靠的基线，再分析每个阶段的耗时。"
        "测试需要固定输入、记录配置，并检查生成语音是否完整，有没有漏字、重复或断音。"
        "每一次修改都应有清晰的目的和可以重复的验证结果。"
    )
    # No gate during warmup, so startup/first-request capture is not counted.
    send_request(args, short, "warmup", False)
    for label, text in [("short", short), ("long", long)]:
        send_request(args, text, label, True)
    args.path = args.trace
    summarize(args)


def summarize(args):
    events = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for begin in [e for e in events if e.get("event") == "client_begin"]:
        ends = [
            e
            for e in events
            if e.get("event") == "client_end"
            and e["label"] == begin["label"]
            and e["time_ns"] >= begin["time_ns"]
        ]
        if not ends:
            continue
        selected = [
            e
            for e in events
            if begin["time_ns"] <= e["time_ns"] <= ends[0]["time_ns"]
            and e.get("event", "").startswith("suffix_")
        ]
        output = {"label": begin["label"]}
        for kind in ["suffix_python", "suffix_graph_replay"]:
            rows = [e for e in selected if e["event"] == kind]
            output[kind] = {
                "calls": len(rows),
                "old_ge_context_calls": sum(
                    e["old_frames"] >= e["context_frames"] for e in rows
                ),
                "old_frames": sorted(set(e["old_frames"] for e in rows)),
                "batch_buckets": sorted(set(e["batch"] for e in rows)),
            }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    print(
        "Counts describe isolated diagnostic requests, not formal performance measurements."
    )
    print(
        "Zero events does not prove zero decoding: check backend choice, source path and outer graph capture."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["inspect", "instrument"]:
        q = sub.add_parser(name)
        q.add_argument(
            "--package-root", help="Directory containing the vllm_omni package contents"
        )
        if name == "instrument":
            q.add_argument("--manifest", type=Path, required=True)
        q.set_defaults(run=inspect_install if name == "inspect" else instrument)
    q = sub.add_parser("restore")
    q.add_argument("--manifest", type=Path, required=True)
    q.set_defaults(run=restore)
    q = sub.add_parser("requests")
    q.add_argument("--url", default="http://127.0.0.1:8000")
    q.add_argument(
        "--model",
        required=True,
        help="Served model name, not necessarily the model path",
    )
    q.add_argument("--voice", default="vivian")
    q.add_argument(
        "--payload",
        type=Path,
        help="Optional known-working raw-audio JSON request template",
    )
    q.add_argument("--trace", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--timeout", type=float, default=600)
    q.set_defaults(run=requests)
    q = sub.add_parser("summarize")
    q.add_argument("path", type=Path)
    q.set_defaults(run=summarize)
    args = p.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
