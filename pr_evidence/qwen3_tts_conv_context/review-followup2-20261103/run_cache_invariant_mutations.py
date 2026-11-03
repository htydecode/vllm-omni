"""Verify the cache invariant assertions detect process-local corruptions."""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mutation", choices=["dtype", "device"], required=True)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    xml = root / ("mutation-" + args.mutation + ".xml")

    class Mutation:
        original = None
        decoder_class = None

        def pytest_collection_modifyitems(self, items):
            import torch
            from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Decoder,
            )
            self.decoder_class = Qwen3TTSTokenizerV2Decoder
            self.original = Qwen3TTSTokenizerV2Decoder.forward
            original = self.original

            def corrupt_cache(decoder, codes, caches=None):
                output = original(decoder, codes, caches=caches)
                if caches is not None and "ref_hidden" in caches:
                    caches["ref_hidden"] = (
                        caches["ref_hidden"].to(torch.float64)
                        if args.mutation == "dtype"
                        else caches["ref_hidden"].to("meta")
                    )
                return output

            Qwen3TTSTokenizerV2Decoder.forward = corrupt_cache

    plugin = Mutation()
    try:
        code = pytest.main([
            "tests/model_executor/models/qwen3_tts/test_qwen3_tts_incremental_decode.py",
            "-k", "test_incremental_request_audio_matches_full_decode", "-m", "core_model and cpu",
            "--run-level=core_model", "-q", "--junitxml=" + str(xml),
            "-o", "cache_dir=" + str(root / ".mutation-cache"),
        ], plugins=[plugin])
    finally:
        if plugin.original is not None:
            plugin.decoder_class.forward = plugin.original
    cases = ET.parse(xml).findall(".//testcase")
    expected = "assert ref_hidden." + args.mutation + " == suffix_quantized." + args.mutation
    detected = int(code) == 1 and len(cases) == 2 and all(
        case.find("failure") is not None and expected in case.find("failure").text for case in cases
    )
    report = {"mutation": args.mutation, "cases": len(cases), "expected_pytest_exit_code": 1,
              "actual_pytest_exit_code": int(code), "expected_assertion": expected,
              "mutation_detected": detected, "production_files_modified": False}
    (root / ("mutation-" + args.mutation + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if detected else 1


if __name__ == "__main__":
    sys.exit(main())
