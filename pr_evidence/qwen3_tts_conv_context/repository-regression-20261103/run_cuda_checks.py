#!/usr/bin/env python3
"""Run the pinned PR's actual repository CPU tests in the prepared Linux env."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

HEAD = "bff6f65ac32fba6739b8327d7caa41996bf7d1b6"
MODEL = "vllm_omni/model_executor/models/qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py"
TEST = "tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py"
RELATED = "tests/model_executor/models/qwen3_tts/test_qwen3_tts_code2wav.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    repo, root = args.source.resolve(), args.result_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", VLLM_TARGET_DEVICE="cuda", VLLM_OMNI_TARGET_DEVICE="cuda",
               PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        env.pop(key, None)
    summary = {"head": HEAD, "backend": "cuda-enabled torch; repository CPU-marked tests", "status": "running", "commands": [], "pytest": {},
               "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def save():
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def run(label, argv):
        argv = list(map(str, argv))
        row = {"label": label, "argv": argv, "exit_code": None}
        summary["commands"].append(row)
        save()
        print("\n$ " + shlex.join(argv), flush=True)
        with (root / (label + ".log")).open("w") as log:
            log.write("$ " + shlex.join(argv) + "\n")
            log.flush()
            with subprocess.Popen(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace") as p:
                for line in p.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                row["exit_code"] = p.wait()
        save()
        return row["exit_code"]

    def junit(label):
        cases = ET.parse(root / (label + ".xml")).findall(".//testcase")
        result = {"tests": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
                  "errors": sum(c.find("error") is not None for c in cases),
                  "skipped": sum(c.find("skipped") is not None for c in cases)}
        result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
        summary["pytest"][label] = result
        save()
        return result

    save()
    try:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != HEAD:
            raise RuntimeError("Unexpected PR head: " + actual)
        expected = {MODEL: "d0b46501c11ff0239970198c71f01e5fcf295d8a6fe0939e1a813b005a70886f",
                    TEST: "648368e2bce9408f3f34fb474099e0b193c35b111df3f4ab34da71e267215a8b"}
        for rel, sha in expected.items():
            if hashlib.sha256((repo / rel).read_bytes()).hexdigest() != sha:
                raise RuntimeError("Source hash mismatch: " + rel)
        metadata = '''import importlib.metadata as m, importlib.util, json, platform, sys
from pathlib import Path
import torch
path = Path(importlib.util.find_spec("vllm_omni").origin).resolve()
assert path.is_relative_to(Path(sys.argv[1]).resolve()), str(path)
versions = {n:m.version(n) for n in ("vllm","vllm-omni","torch","transformers","pytest","pytest-mock","pytest-asyncio","pytest-xdist")}
assert versions["vllm"].split("+")[0] == "0.32.0", versions
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available() and torch.backends.cudnn.is_available()
data = {"python":sys.version,"platform":platform.platform(),"libc":platform.libc_ver(),
        "versions":versions,"backend":"cuda-enabled","gpu":torch.cuda.get_device_name(0),"cudnn_version":torch.backends.cudnn.version(),"cuda_runtime":torch.version.cuda,"source":str(path)}
Path(sys.argv[2]).write_text(json.dumps(data,indent=2)+"\\n")
print(json.dumps(data,indent=2))
'''
        if run("environment", [sys.executable, "-c", metadata, repo, root / "environment.json"]):
            raise RuntimeError("Environment validation failed")
        common = ["-m", "core_model and cpu", "--run-level=core_model", "-q",
                  "-o", "cache_dir=" + str(root / ".pytest_cache")]
        rc = run("targeted-pytest", [sys.executable, "-m", "pytest", TEST, "-k",
                 "suffix_pre_conv_context_avoids_full_history_cat", *common,
                 "--junitxml=" + str(root / "targeted-pytest.xml")])
        counts = junit("targeted-pytest")
        if rc or counts != {"tests":120,"passed":120,"failures":0,"errors":0,"skipped":0}:
            raise RuntimeError("Expected 120 passed and no errors/failures/skips")
        rc = run("related-pytest", [sys.executable, "-m", "pytest", TEST, RELATED, *common,
                 "--junitxml=" + str(root / "related-pytest.xml")])
        counts = junit("related-pytest")
        if rc or counts["errors"] or counts["failures"] or not counts["passed"]:
            raise RuntimeError("Related repository regression failed")
        summary["status"] = "passed"
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        (root / "runner-error.log").write_text(traceback.format_exc())
        print("FAILED: " + str(exc), flush=True)
    finally:
        summary["finish_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print("Local CUDA-enabled repository regression: " + summary["status"], flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
