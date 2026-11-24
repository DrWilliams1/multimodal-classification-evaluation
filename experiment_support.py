"""Data, CLIP inference, and export helpers for the teaching notebook.

No training is performed. Test labels never enter prompt or threshold selection.
"""
import csv
import hashlib
import io
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
import numpy as np
from multimodal_utils import (softmax, classification_metrics, choose_method,
    coverage_threshold, selective_metrics, paired_accuracy_bootstrap)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode_table(path):
    """Read pinned Parquet directly; do not execute a dataset loading script."""
    import pyarrow.parquet as pq
    from PIL import Image
    table = pq.read_table(path)
    labels = np.asarray(table['label'].to_pylist(), dtype=np.int64)
    column = 'img' if 'img' in table.column_names else 'image'
    images = []
    for item in table[column].to_pylist():
        with Image.open(io.BytesIO(item['bytes'])) as img:
            pixels = np.asarray(img.convert('RGB'), dtype=np.uint8)
            if pixels.shape != (32, 32, 3):
                raise ValueError('Unexpected CIFAR-10 image dimensions')
            images.append(pixels)
    return np.stack(images), labels


def load_data(config):
    """Download immutable source revisions and audit exact RGB duplicates."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    arrays, source = {}, {}
    for split, expected in [('train', 50000), ('test', 10000)]:
        relative = f'plain_text/{split}-00000-of-00001.parquet'
        path = hf_hub_download(config['dataset_id'], relative, repo_type='dataset',
                               revision=config['dataset_revision'])
        source_sha256 = file_hash(path)
        if source_sha256 != config['dataset_file_sha256'][split]:
            raise ValueError('Pinned dataset file hash differs')
        metadata = json.loads(pq.read_schema(path).metadata[b'huggingface'])
        if metadata['info']['features']['label']['names'] != config['class_names']:
            raise ValueError('Dataset class order differs from the protocol')
        images, labels = decode_table(path)
        if len(labels) != expected or not np.array_equal(np.bincount(labels, minlength=10),
                                                         np.full(10, expected // 10)):
            raise ValueError('Unexpected split size or class distribution')
        hashes = np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in images])
        arrays[split] = {'images': images, 'labels': labels, 'hashes': hashes}
        source[split] = {'path_in_repository': relative, 'sha256': source_sha256,
                         'n_examples': len(labels)}
    train, test = arrays['train'], arrays['test']
    test_hashes = set(test['hashes'])
    seen = set()
    eligible = []
    cross_split_excluded = within_train_excluded = 0
    for i, h in enumerate(train['hashes']):
        if h in test_hashes:
            cross_split_excluded += 1
        elif h in seen:
            within_train_excluded += 1
        else:
            eligible.append(i)
            seen.add(h)
    audit = {
        'dataset_id': config['dataset_id'], 'revision': config['dataset_revision'],
        'source_files': source, 'hash_definition': 'SHA256 of decoded 32x32x3 RGB uint8 bytes',
        'train_rows_sharing_exact_pixels_with_test_excluded': cross_split_excluded,
        'additional_duplicate_train_rows_excluded': within_train_excluded,
        'eligible_training_rows': len(eligible),
        'duplicate_test_rows_beyond_first': len(test['hashes']) - len(test_hashes),
        'test_policy': 'Keep every official test row; report exact duplicates; no near-duplicate audit.',
        'pretraining_exposure': 'Unknown; zero-shot here means no task-specific weight fitting.'
    }
    return arrays, np.asarray(eligible, dtype=np.int64), audit


def sample_splits(labels, eligible, selection_per_class, threshold_per_class, seed):
    """Disjoint balanced samples; fixed RNG and original-row IDs are retained."""
    rng = np.random.default_rng(seed)
    dev, calibration = [], []
    n = selection_per_class + threshold_per_class
    for label in range(10):
        pool = eligible[labels[eligible] == label]
        if len(pool) < n:
            raise ValueError('Not enough eligible images in a class')
        selected = rng.choice(pool, size=n, replace=False)
        dev.extend(selected[:selection_per_class])
        calibration.extend(selected[selection_per_class:])
    return np.sort(dev).astype(np.int64), np.sort(calibration).astype(np.int64)


def load_clip(config, device):
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
    torch.set_num_threads(config['torch_threads'])
    # Explicit eager attention and float32 reduce implementation ambiguity.
    model = CLIPModel.from_pretrained(config['model_id'], revision=config['model_revision'],
        use_safetensors=False, attn_implementation='eager').to(device).float().eval()
    model.requires_grad_(False)
    tokenizer = CLIPTokenizerFast.from_pretrained(config['model_id'], revision=config['model_revision'])
    processor = CLIPImageProcessor.from_pretrained(config['model_id'], revision=config['model_revision'])
    return model, tokenizer, processor


def build_text_prototypes(model, tokenizer, config, device):
    """Normalize each text vector, average by class, then normalize again."""
    import torch
    prompts, prototypes = {}, {}
    with torch.inference_mode():
        for method in config['method_order']:
            by_class, texts = [], []
            for label in config['class_names']:
                article = 'an' if label in ('airplane', 'automobile') else 'a'
                group = [t.format(label=label, article=article) for t in config['templates'][method]]
                texts.extend(group)
                by_class.append(group)
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(device)
            if inputs['input_ids'].shape[1] > model.config.text_config.max_position_embeddings:
                raise ValueError('A prompt exceeds the model token limit')
            features = model.get_text_features(**inputs).float()
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.reshape(10, len(config['templates'][method]), -1).mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
            prototypes[method] = features.cpu().numpy()
            prompts[method] = by_class
    return prototypes, prompts


def encode_images(model, processor, images, config, device, description='Images'):
    """One encoder pass per image; every prompt method reuses the same vector."""
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    vectors = []
    if device == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in tqdm(range(0, len(images), config['batch_size']), desc=description):
            batch = [Image.fromarray(x) for x in images[start:start+config['batch_size']]]
            inputs = processor(images=batch, return_tensors='pt').to(device)
            features = model.get_image_features(**inputs).float()
            features = features / features.norm(dim=-1, keepdim=True)
            vectors.append(features.cpu().numpy())
    if device == 'cuda':
        torch.cuda.synchronize()
    return np.concatenate(vectors), time.perf_counter() - started


def compute_logits(image_vectors, prototypes, logit_scale, method_order):
    # Shape: method x image x class. Use float64 scoring for stable saved metrics.
    return np.stack([logit_scale * image_vectors.astype(np.float64) @ prototypes[m].astype(np.float64).T
                     for m in method_order])


def summarize_split(y_true, logits, methods):
    return {m: classification_metrics(y_true, softmax(logits[i])) for i, m in enumerate(methods)}


def lock_protocol(config, dev_labels, dev_logits, threshold_logits):
    """No test data is accepted by this function."""
    metrics = summarize_split(dev_labels, dev_logits, config['method_order'])
    selected = choose_method(metrics, config['method_order'])
    idx = config['method_order'].index(selected)
    confidence = softmax(threshold_logits[idx]).max(axis=1)
    thresholds = {str(t): coverage_threshold(confidence, t) for t in config['coverage_targets']}
    return {'selected_method': selected, 'selection_metric': 'development top-1 accuracy',
            'tie_priority': config['method_order'], 'thresholds': thresholds,
            'threshold_rule': 'kth-largest calibration score, k=ceil(target*n); accept score>=threshold',
            'primary_coverage_target': config['primary_coverage_target'],
            'scores_are_calibrated_probabilities': False,
            'locked_before_test_inference': True}


def evaluate_records(records, config, protocol):
    methods = config['method_order']
    metrics = {split: summarize_split(records[f'{split}_labels'], records[f'{split}_logits'], methods)
               for split in ('development', 'threshold', 'test')}
    index = methods.index(protocol['selected_method'])
    test_prob = softmax(records['test_logits'][index])
    selective = {target: selective_metrics(records['test_labels'], test_prob, threshold)
                 for target, threshold in protocol['thresholds'].items()}
    comparisons = {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        comparisons[f'{methods[b]}_minus_{methods[a]}'] = paired_accuracy_bootstrap(
            records['test_labels'], softmax(records['test_logits'][a]), softmax(records['test_logits'][b]),
            resamples=config['bootstrap_resamples'], seed=config['seed'])
    return metrics, selective, comparisons


def environment_record(device):
    import importlib.metadata as meta
    import torch
    names = ['torch','transformers','huggingface-hub','numpy','pandas','pyarrow','Pillow','matplotlib']
    return {'python': platform.python_version(), 'platform': platform.platform(), 'device': device,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'versions': {name: meta.version(name) for name in names},
            'torch_threads': torch.get_num_threads(), 'precision': 'float32 encoder, float64 score/metric arithmetic'}


def export_results(out, config, records, protocol, audit, prompts, prototypes, timings, device):
    import pandas as pd
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, selective, comparisons = evaluate_records(records, config, protocol)
    for name, value in [('config', config), ('protocol', protocol), ('data_audit', audit), ('prompts', prompts),
                        ('metrics', metrics), ('selective_metrics', selective), ('paired_comparisons', comparisons),
                        ('timing', timings), ('environment', environment_record(device))]:
        write_json(out / f'{name}.json', value)
    np.savez_compressed(out / 'records.npz', **records)
    np.savez_compressed(out / 'text_prototypes.npz', **prototypes)
    rows, summary = [], []
    for split in ('development', 'threshold', 'test'):
        labels, ids = records[f'{split}_labels'], records[f'{split}_ids']
        for index, method in enumerate(config['method_order']):
            prob = softmax(records[f'{split}_logits'][index])
            pred = prob.argmax(axis=1)
            for k in range(len(labels)):
                rows.append({'split':split, 'row_id':int(ids[k]), 'method':method,
                             'actual_label':int(labels[k]), 'predicted_label':int(pred[k]),
                             'max_softmax_score':float(prob[k].max()), 'correct':bool(pred[k] == labels[k])})
            value = metrics[split][method]
            summary.append({'split':split, 'method':method, 'n_examples':value['n_examples'],
                            'accuracy':value['accuracy'], 'macro_f1':value['macro_f1']})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'metrics_summary.csv', index=False)
    pd.DataFrame([{'calibration_target_coverage':float(k), **{x:y for x,y in v.items() if not isinstance(y,list)}}
                  for k,v in selective.items()]).to_csv(out / 'selective_summary.csv', index=False)
    errors = predictions[(predictions['split']=='development') &
                         (predictions['method']==protocol['selected_method']) & (~predictions['correct'])]
    errors = errors.sort_values(['max_softmax_score','row_id'], ascending=[False,True]).head(10).copy()
    errors['my_observation'] = ''
    errors.to_csv(out / 'error_analysis.csv', index=False)
    write_json(out / 'run_status.json', {'status':'complete', 'smoke_only':config['smoke_only'],
        'test_evaluated':len(records['test_labels']), 'human_review_outcomes_measured':False})
    return metrics, selective, comparisons, predictions


def plot_results(out, config, metrics, selective, protocol):
    import matplotlib.pyplot as plt
    out = Path(out)
    methods = config['method_order']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(['Labels','Photo prompt','Ensemble'], [metrics['test'][m]['accuracy'] for m in methods], color=['#516880','#207e90','#c27c32'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Test accuracy')
    axes[0].set_title('Fixed prompt comparison')
    points = sorted(selective.values(), key=lambda x:x['coverage'])
    axes[1].plot([p['coverage'] for p in points], [p['selective_accuracy'] for p in points], 'o-', color='#207e90')
    axes[1].scatter([1], [metrics['test'][protocol['selected_method']]['accuracy']], marker='s', color='#516880', label='All predictions')
    axes[1].set(xlim=(0,1.03), ylim=(0,1.03), xlabel='Fraction of test images accepted', ylabel='Accuracy among accepted images', title='Thresholds fixed on separate calibration images')
    axes[1].legend()
    fig.suptitle('CLIP prompt sensitivity and abstention — recorded run')
    fig.tight_layout()
    fig.savefig(out/'accuracy_coverage.png', dpi=170)
    plt.show()
    plt.close(fig)


def finalize_manifest(out):
    out = Path(out)
    files = {str(p.relative_to(out)):file_hash(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name != 'manifest_sha256.json'}
    write_json(out/'manifest_sha256.json', files)
