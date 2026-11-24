"""Run the notebook experiment from a terminal; --smoke uses 60 training images only."""
import argparse
from experiment_support import *
parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
SMOKE_ONLY=args.smoke
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv, hashlib, io, json, math, platform, shutil, time, uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from IPython.display import display

CONFIG = json.loads('{\n  "experiment": "clip_prompt_abstention",\n  "seed": 2026,\n  "model_id": "openai/clip-vit-base-patch32",\n  "model_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",\n  "dataset_id": "uoft-cs/cifar10",\n  "dataset_revision": "0b2714987fa478483af9968de7c934580d0bb9a2",\n  "class_names": [\n    "airplane",\n    "automobile",\n    "bird",\n    "cat",\n    "deer",\n    "dog",\n    "frog",\n    "horse",\n    "ship",\n    "truck"\n  ],\n  "method_order": [\n    "label_only",\n    "photo_prompt",\n    "prompt_ensemble"\n  ],\n  "templates": {\n    "label_only": [\n      "{label}"\n    ],\n    "photo_prompt": [\n      "a photo of {article} {label}."\n    ],\n    "prompt_ensemble": [\n      "a photo of {article} {label}.",\n      "a blurry photo of {article} {label}.",\n      "a close-up photo of {article} {label}.",\n      "a bright photo of {article} {label}.",\n      "a cropped photo of {article} {label}."\n    ]\n  },\n  "selection_per_class": 100,\n  "threshold_per_class": 100,\n  "coverage_targets": [\n    0.5,\n    0.75,\n    0.9\n  ],\n  "primary_coverage_target": 0.75,\n  "bootstrap_resamples": 2000,\n  "batch_size": 64,\n  "torch_threads": 4,\n  "precision": "float32",\n  "smoke_only": false,\n  "dataset_file_sha256": {\n    "train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",\n    "test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"\n  }\n}')
# A CLI preparation check can set SMOKE_ONLY; the Colab notebook uses False.
SMOKE_ONLY = bool(globals().get("SMOKE_ONLY", False))
CONFIG["smoke_only"] = SMOKE_ONLY
if SMOKE_ONLY:
    CONFIG["selection_per_class"] = 2
    CONFIG["threshold_per_class"] = 2
    CONFIG["test_source_split"] = "unused_training_rows_for_smoke_only"
else:
    CONFIG["test_source_split"] = "official_test"

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.set_num_threads(CONFIG["torch_threads"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
if not SMOKE_ONLY and DEVICE != "cuda":
    raise RuntimeError("Choose a Colab T4 GPU, then rerun from this cell.")
RUN_ID = ("smoke_" if SMOKE_ONLY else "multimodal_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
OUT = Path("artifacts") / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
print("Run:", RUN_ID, "| device:", DEVICE, "| smoke only:", SMOKE_ONLY)


started = time.perf_counter()
DATA, ELIGIBLE, AUDIT = load_data(CONFIG)
TIMINGS = {"data_download_decode_audit_seconds": time.perf_counter() - started}
train = DATA["train"]
if SMOKE_ONLY:
    # Use only training images during preparation; do not expose official test predictions.
    dev_ids, other_ids = sample_splits(train["labels"], ELIGIBLE, 2, 4, CONFIG["seed"])
    cal_ids, test_ids = [], []
    for label in range(10):
        candidates = other_ids[train["labels"][other_ids] == label]
        cal_ids.extend(candidates[:2])
        test_ids.extend(candidates[2:])
    cal_ids, test_ids = np.sort(cal_ids), np.sort(test_ids)
    TEST_SOURCE = train
else:
    dev_ids, cal_ids = sample_splits(train["labels"], ELIGIBLE,
        CONFIG["selection_per_class"], CONFIG["threshold_per_class"], CONFIG["seed"])
    test_ids = np.arange(len(DATA["test"]["labels"]))
    TEST_SOURCE = DATA["test"]

assert not set(dev_ids) & set(cal_ids)
RECORDS = {}
for split, source, ids in [("development", train, dev_ids), ("threshold", train, cal_ids),
                           ("test", TEST_SOURCE, test_ids)]:
    RECORDS[f"{split}_ids"] = np.asarray(ids, dtype=np.int64)
    RECORDS[f"{split}_labels"] = source["labels"][ids]
    RECORDS[f"{split}_pixel_hashes"] = source["hashes"][ids]
write_json(OUT/"config.json", CONFIG)
write_json(OUT/"data_audit.json", AUDIT)
print(json.dumps(AUDIT, indent=2))
print({s:len(RECORDS[f"{s}_ids"]) for s in ("development","threshold","test")})

started = time.perf_counter()
MODEL, TOKENIZER, PROCESSOR = load_clip(CONFIG, DEVICE)
TIMINGS["model_loading_seconds"] = time.perf_counter() - started
write_json(OUT/"preprocessing.json", PROCESSOR.to_dict())
started = time.perf_counter()
PROTOTYPES, PROMPTS = build_text_prototypes(MODEL, TOKENIZER, CONFIG, DEVICE)
TIMINGS["text_prototype_seconds"] = time.perf_counter() - started
LOGIT_SCALE = float(MODEL.logit_scale.exp().detach().cpu())
TIMINGS["fixed_logit_scale"] = LOGIT_SCALE
write_json(OUT/"prompts.json", PROMPTS)
for method in CONFIG["method_order"]:
    print(method, "— airplane prompts:", PROMPTS[method][0])
    assert np.allclose(np.linalg.norm(PROTOTYPES[method], axis=1), 1, atol=1e-5)

for split, ids in [("development", dev_ids), ("threshold", cal_ids)]:
    vectors, elapsed = encode_images(MODEL, PROCESSOR, train["images"][ids], CONFIG, DEVICE, split)
    TIMINGS[f"{split}_image_encoding_seconds"] = elapsed
    RECORDS[f"{split}_logits"] = compute_logits(vectors, PROTOTYPES, LOGIT_SCALE, CONFIG["method_order"])

PROTOCOL = lock_protocol(CONFIG, RECORDS["development_labels"],
    RECORDS["development_logits"], RECORDS["threshold_logits"])
write_json(OUT/"protocol.json", PROTOCOL)
LOCKED_PROTOCOL_SHA256 = file_hash(OUT/"protocol.json")
print(json.dumps(PROTOCOL, indent=2))
display(pd.DataFrame(summarize_split(RECORDS["development_labels"],
    RECORDS["development_logits"], CONFIG["method_order"])).T[["accuracy","macro_f1","n_errors"]])

selected_index = CONFIG["method_order"].index(PROTOCOL["selected_method"])
cal_scores = softmax(RECORDS["threshold_logits"][selected_index]).max(axis=1)
print("Calibration coverage, including threshold ties:")
print({target:float(np.mean(cal_scores >= cutoff)) for target,cutoff in PROTOCOL["thresholds"].items()})

assert file_hash(OUT/"protocol.json") == LOCKED_PROTOCOL_SHA256
vectors, elapsed = encode_images(MODEL, PROCESSOR, TEST_SOURCE["images"][test_ids], CONFIG, DEVICE, "test" if not SMOKE_ONLY else "smoke training images")
TIMINGS["test_image_encoding_seconds"] = elapsed
started = time.perf_counter()
RECORDS["test_logits"] = compute_logits(vectors, PROTOTYPES, LOGIT_SCALE, CONFIG["method_order"])
TIMINGS["all_test_prompt_scoring_seconds"] = time.perf_counter() - started
TIMINGS["interpretation"] = "Setup and batched encoding/scoring; not per-request latency; download cache state varies."
assert file_hash(OUT/"protocol.json") == LOCKED_PROTOCOL_SHA256
METRICS, SELECTIVE, COMPARISONS, PREDICTIONS = export_results(OUT, CONFIG, RECORDS,
    PROTOCOL, AUDIT, PROMPTS, PROTOTYPES, TIMINGS, DEVICE)
assert file_hash(OUT/"protocol.json") == LOCKED_PROTOCOL_SHA256
display(pd.read_csv(OUT/"metrics_summary.csv").query("split == 'test'"))
display(pd.read_csv(OUT/"selective_summary.csv"))
print(json.dumps(COMPARISONS, indent=2))

plot_results(OUT, CONFIG, METRICS, SELECTIVE, PROTOCOL)
primary = SELECTIVE[str(CONFIG["primary_coverage_target"])]
print("Selected method:", PROTOCOL["selected_method"])
print("At the primary operating point:", json.dumps(primary, indent=2))
display(pd.DataFrame({"class":CONFIG["class_names"],
    "coverage":primary["per_class_coverage"],
    "accepted_accuracy":primary["per_class_accepted_accuracy"]}))
print("Development errors for inspection:")
display(pd.read_csv(OUT/"error_analysis.csv"))

finalize_manifest(OUT)
ZIP_PATH = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
print("Results:", OUT.resolve())
print("ZIP:", Path(ZIP_PATH).resolve())
if not SMOKE_ONLY:
    try:
        from google.colab import files
        files.download(ZIP_PATH)
    except ImportError:
        print("Outside Colab: download the ZIP from the notebook file browser.")
