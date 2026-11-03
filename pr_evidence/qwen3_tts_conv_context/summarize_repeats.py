"""Summarize independent runs of the existing local A/B benchmark."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    identities = set()
    paths = [
        p
        for p in sorted(args.directory.glob("*.json"))
        if p.resolve() != args.output.resolve()
    ]
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        required = [
            "torch",
            "gpu",
            "dtype",
            "channels",
            "baseline_sha256",
            "candidate_sha256",
        ]
        if not all(k in report for k in required) or not report.get("results"):
            raise RuntimeError(
                f"Not a local A/B report: {path}; use a dedicated result directory"
            )
        identities.add(tuple(report[k] for k in required))
        for row in report["results"]:
            key = (
                row["mode"],
                row["batch"],
                row["ref_frames"],
                row["old_frames"],
                row["new_frames"],
            )
            before, after = row["samples"]["baseline"], row["samples"]["candidate"]
            if len(before) != len(after) or not before:
                raise RuntimeError(f"Unpaired samples: {path}")
            differences = [
                a["wall_us_per_call"] - b["wall_us_per_call"]
                for a, b in zip(before, after)
            ]
            groups.setdefault(key, []).append(
                {
                    "file": str(path),
                    "paired_median_saved_us": statistics.median(differences),
                }
            )
    if not paths or len(identities) != 1:
        raise RuntimeError(
            "Need at least one report, with matching device/software/shape/source identity"
        )
    rows = []
    print("mode batch ref old runs positive median_saved_us min_saved_us max_saved_us")
    for key, samples in sorted(groups.items()):
        differences = [s["paired_median_saved_us"] for s in samples]
        row = {
            "case": list(key),
            "independent_runs": samples,
            "positive_runs": sum(x > 0 for x in differences),
            "median_saved_us": statistics.median(differences),
            "min_saved_us": min(differences),
            "max_saved_us": max(differences),
        }
        rows.append(row)
        print(
            *key[:4],
            len(samples),
            row["positive_runs"],
            f"{row['median_saved_us']:.3f}",
            f"{row['min_saved_us']:.3f}",
            f"{row['max_saved_us']:.3f}",
        )
    output = {
        "source_reports": [str(p) for p in paths],
        "results": rows,
        "interpretation": "Independent-run paired medians. Min/max is not a confidence interval.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
