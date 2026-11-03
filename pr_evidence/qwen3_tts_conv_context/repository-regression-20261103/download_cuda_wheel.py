#!/usr/bin/env python3
"""Download the official pinned CPU wheels in verified HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ASSETS = [("vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl", "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl", 326518482, "5e2ecdc5b0baf72921c288486417ca82778456a448e4fe60b62ee43c0dfc8611")]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root, name, url, size, expected):
    final = root / name
    if final.is_file() and final.stat().st_size == size and sha256(final) == expected:
        print("Verified existing " + name, flush=True)
        return
    partial = root / (name + ".partial")
    resume = partial.is_file() and partial.stat().st_size == size
    with partial.open("r+b" if resume else "w+b") as f:
        f.truncate(size)
        fd = f.fileno()

        def part(start):
            end = min(start + 1024 * 1024, size) - 1
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url + f"?range={start}&retry={time.time_ns()}", headers={
                        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError("Server did not honor exact byte range")
                        data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete range")
                    view = memoryview(data)
                    offset = start
                    while view:
                        written = os.pwrite(fd, view, offset)
                        if written <= 0:
                            raise RuntimeError("Partial file write failed")
                        offset += written
                        view = view[written:]
                    return len(data)
                except Exception:
                    if attempt == 3:
                        raise RuntimeError(f"Range {start}-{end} failed after retries") from None
                    time.sleep(1 + attempt)
        starts = []
        done = 0
        for start in range(0, size, 1024 * 1024):
            length = min(1024 * 1024, size - start)
            if resume and any(os.pread(fd, length, start)):
                done += length
            else:
                starts.append(start)
        print(f"Resuming {name}: {done / 1048576:.1f} MiB present; final SHA verification required", flush=True)
        with ThreadPoolExecutor(max_workers=24) as pool:
            futures = [pool.submit(part, start) for start in starts]
            for i, future in enumerate(as_completed(futures), 1):
                done += future.result()
                if i % 16 == 0 or done == size:
                    print(f"{name}: {done / 1048576:.1f}/{size / 1048576:.1f} MiB", flush=True)
        f.flush()
        os.fsync(fd)
    if sha256(partial) != expected:
        raise RuntimeError("Official SHA-256 mismatch: " + name)
    partial.replace(final)
    print("SHA-256 verified: " + name, flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        download(root, *asset)
    (root / "official-assets.json").write_text(json.dumps([
        {"name": n, "url": u, "size": s, "sha256": h} for n, u, s, h in ASSETS
    ], indent=2) + "\n")
