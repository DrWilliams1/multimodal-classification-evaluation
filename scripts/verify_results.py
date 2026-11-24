"""Verify a saved run without loading images, models, pandas, or PyTorch.

This checks internal consistency and file integrity. It does not establish
source authenticity, reproduce neural inference, or prove execution chronology.
Run: python scripts/verify_results.py path/to/run_directory
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_support import evaluate_records, lock_protocol
from multimodal_utils import softmax


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, path="value"):
    """Compare saved JSON with independently recomputed records."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{path}: list length differs")
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, f"{path}: differs")
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, f"{path}: integer differs")
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), f"{path}: number differs")
    else:
        raise TypeError(f"Unsupported comparison type at {path}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def verify_manifest(root):
    manifest = read_json(root / "manifest_sha256.json")
    require(isinstance(manifest, dict) and bool(manifest), "Empty or malformed manifest")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "manifest_sha256.json"}
    require(actual == set(manifest), "Manifest file list does not match directory")
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), "Manifest path leaves run directory")
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Invalid manifest digest")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == expected, f"SHA256 mismatch: {relative}")
    return len(manifest)


def verify_csv(path, expected_rows, key_columns):
    expected_rows = list(expected_rows)
    require(bool(expected_rows), f"{path.name}: no expected rows")
    columns = set(expected_rows[0])
    expected = {tuple(str(row[k]) for k in key_columns): row for row in expected_rows}
    require(len(expected) == len(expected_rows), f"{path.name}: duplicate expected keys")
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(columns)
                and set(reader.fieldnames) == columns, f"{path.name}: columns differ")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),
                    f"{path.name}: malformed CSV row")
            key = tuple(row[k] for k in key_columns)
            require(key in expected and key not in seen, f"{path.name}: unexpected or repeated row {key}")
            seen.add(key)
            for field, value in expected[key].items():
                cell = row[field]
                location = f"{path.name}/{key}/{field}"
                if value is None:
                    require(cell == "", f"{location}: expected blank")
                elif isinstance(value, bool):
                    require(cell in ("True", "False") and (cell == "True") == value, f"{location}: boolean differs")
                elif isinstance(value, int):
                    compare(int(cell), value, location)
                elif isinstance(value, float):
                    compare(float(cell), value, location)
                else:
                    compare(cell, value, location)
    require(seen == set(expected), f"{path.name}: missing rows")
    return len(seen)


def prediction_rows(records, methods):
    for split in ("development", "threshold", "test"):
        labels, ids = records[f"{split}_labels"], records[f"{split}_ids"]
        for index, method in enumerate(methods):
            probabilities = softmax(records[f"{split}_logits"][index])
            predictions = probabilities.argmax(axis=1)
            for i in range(len(labels)):
                yield {"split": split, "row_id": int(ids[i]), "method": method,
                       "actual_label": int(labels[i]), "predicted_label": int(predictions[i]),
                       "max_softmax_score": float(probabilities[i].max()),
                       "correct": bool(predictions[i] == labels[i])}


def verify(root):
    root = root.resolve()
    hashed_files = verify_manifest(root)
    names = ("config", "protocol", "metrics", "selective_metrics", "paired_comparisons",
             "data_audit", "prompts", "run_status")
    data = {name: read_json(root / f"{name}.json") for name in names}
    config, protocol, audit, status = (data[k] for k in ("config", "protocol", "data_audit", "run_status"))
    methods = config["method_order"]
    require(methods == ["label_only", "photo_prompt", "prompt_ensemble"], "Unexpected method order")
    require(config["class_names"] == ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "Unexpected class labels or order")
    require(type(config["smoke_only"]) is bool, "smoke_only must be a boolean")
    smoke = config["smoke_only"]
    require(status["status"] == "complete" and status["smoke_only"] is smoke, "Incomplete or inconsistent run status")
    require(status["human_review_outcomes_measured"] is False, "Unsupported human review outcome claim")
    require(audit["dataset_id"] == config["dataset_id"] and audit["revision"] == config["dataset_revision"],
            "Dataset identifiers differ")
    for split, count in (("train", 50000), ("test", 10000)):
        source = audit["source_files"][split]
        require(source["n_examples"] == count, "Unexpected source split size")
        require(re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, "Invalid source hash format")
        if "dataset_file_sha256" in config:
            require(source["sha256"] == config["dataset_file_sha256"][split], "Source hash differs from fixed configuration")
    audit_counts = [audit[k] for k in ("train_rows_sharing_exact_pixels_with_test_excluded",
                    "additional_duplicate_train_rows_excluded", "eligible_training_rows")]
    require(all(type(v) is int and v >= 0 for v in audit_counts) and sum(audit_counts) == 50000,
            "Training exclusion counts do not reconcile")
    require(type(audit["duplicate_test_rows_beyond_first"]) is int
            and 0 <= audit["duplicate_test_rows_beyond_first"] < 10000, "Invalid test duplicate count")
    with np.load(root / "records.npz", allow_pickle=False) as archive:
        records = {name: archive[name] for name in archive.files}
    counts = {}
    pixel_hash_sets = {}
    for split in ("development", "threshold", "test"):
        ids, labels, logits, hashes = (records[f"{split}_{suffix}"] for suffix in
                                       ("ids", "labels", "logits", "pixel_hashes"))
        n = len(labels)
        counts[split] = n
        require(n > 0 and labels.shape == (n,) and labels.dtype.kind in "iu"
                and np.all((labels >= 0) & (labels < 10)), f"{split}: malformed labels")
        require(ids.shape == (n,) and ids.dtype.kind in "iu" and len(np.unique(ids)) == n
                and np.all((ids >= 0) & (ids < (50000 if split != "test" or smoke else 10000))),
                f"{split}: malformed or repeated row IDs")
        require(logits.shape == (len(methods), n, 10) and logits.dtype.kind in "fiu"
                and np.isfinite(logits).all(), f"{split}: malformed logits")
        require(hashes.shape == (n,) and hashes.dtype.kind == "U"
                and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes), f"{split}: malformed pixel hashes")
        pixel_hash_sets[split] = set(hashes.tolist())
        if split != "test":
            require(len(pixel_hash_sets[split]) == n, f"{split}: repeated pixel hashes")
            per_class = config["selection_per_class" if split == "development" else "threshold_per_class"]
            require(type(per_class) is int and per_class > 0
                    and np.array_equal(np.bincount(labels, minlength=10), np.full(10, per_class)),
                    f"{split}: class counts differ from configured sample")
        if not smoke:
            expected_per_class = 1000 if split == "test" else 100
            require(np.array_equal(np.bincount(labels, minlength=10), np.full(10, expected_per_class)),
                    f"{split}: full-run class counts are wrong")
    require(not set(records["development_ids"]) & set(records["threshold_ids"]), "Training-derived splits overlap")
    for a, b in (("development", "threshold"), ("development", "test"), ("threshold", "test")):
        require(not pixel_hash_sets[a] & pixel_hash_sets[b], f"Pixel hashes overlap: {a}, {b}")
    if not smoke:
        require(np.array_equal(np.sort(records["test_ids"]), np.arange(10000)), "Official test rows are missing")
        require(counts["test"] - len(pixel_hash_sets["test"]) == audit["duplicate_test_rows_beyond_first"],
                "Saved test hashes disagree with reported duplicates")
    require(status["test_evaluated"] == counts["test"], "Status test count differs")
    expected_prompts = {method: [[template.format(label=label,
                       article="an" if label in ("airplane", "automobile") else "a")
                       for template in config["templates"][method]] for label in config["class_names"]]
                       for method in methods}
    compare(data["prompts"], expected_prompts, "prompts")
    with np.load(root / "text_prototypes.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(methods), "Prototype method keys differ")
        for method in methods:
            vectors = archive[method]
            require(vectors.shape == (10, 512) and vectors.dtype.kind == "f"
                    and np.isfinite(vectors).all()
                    and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5),
                    f"{method}: malformed or non-unit prototypes")
    locked = lock_protocol(config, records["development_labels"], records["development_logits"], records["threshold_logits"])
    compare(protocol, locked, "protocol")
    metrics, selective, comparisons = evaluate_records(records, config, locked)
    for name, value in (("metrics", metrics), ("selective_metrics", selective), ("paired_comparisons", comparisons)):
        compare(data[name], value, name)
    prediction_count = verify_csv(root / "predictions.csv", prediction_rows(records, methods), ("split", "method", "row_id"))
    summary = [{"split": split, "method": method, **{k: value[k] for k in ("n_examples", "accuracy", "macro_f1")}}
               for split, by_method in metrics.items() for method, value in by_method.items()]
    verify_csv(root / "metrics_summary.csv", summary, ("split", "method"))
    selective_rows = [{"calibration_target_coverage": float(target),
                       **{k: v for k, v in values.items() if not isinstance(v, list)}}
                      for target, values in selective.items()]
    verify_csv(root / "selective_summary.csv", selective_rows, ("calibration_target_coverage",))
    return {"status": "passed", "smoke_only": smoke, "hashed_files": hashed_files,
            "examples_by_split": counts, "prediction_rows_checked": prediction_count,
            "selected_method": locked["selected_method"], "model_inference_repeated": False,
            "scope": "Saved-record consistency and file integrity; source authenticity and execution chronology not independently verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_directory)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
