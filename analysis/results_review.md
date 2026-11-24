# Results review

Sources: the recorded run `multimodal_20261124T171506Z-2512b9`, specifically `metrics_summary.csv`, `selective_summary.csv`, `selective_metrics.json`, `paired_comparisons.json`, `protocol.json`, and `prompts.json`. Percentages below are rounded for presentation; counts and stored metrics remain authoritative.

## Prompt comparison and review thresholds

The experiment compared three ways to describe the same ten image categories to a frozen CLIP model. On the 1,000-image development set, the grammatical photo prompt achieved **89.1% accuracy**, narrowly ahead of the five-template ensemble at **89.0%** and bare class labels at **88.5%**. The development rule therefore selected `photo_prompt` for the review-routing experiment before test inference.

| Prompt method | Test accuracy | Test macro F1 | Correct / 10,000 |
| --- | ---: | ---: | ---: |
| Class label only | 87.35% | 0.8734 | 8,735 |
| Photo prompt — selected on development | 88.23% | 0.8817 | 8,823 |
| Five-template ensemble | 88.54% | 0.8847 | 8,854 |

The ensemble had the highest test accuracy, but this does not change the development-selected method. Relative to bare labels, photo prompts correctly classified 88 additional test images and the ensemble correctly classified 119 additional images. The ensemble's advantage over the selected photo prompt was 31 images, or 0.31 percentage points.

The next experiment used the selected method's maximum softmax score to separate automatically accepted predictions from predictions referred for review. Cutoffs were fixed on a separate 1,000-image threshold-setting sample. These scores are **uncalibrated**: a score of 0.75 should not be interpreted as a 75% probability of correctness.

| Target coverage on threshold sample | Score cutoff | Test images accepted | Test images referred | Accuracy among accepted | Errors still accepted | Original errors routed for review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 0.935326 | 4,912 (49.12%) | 5,088 (50.88%) | 99.25% | 37 | 1,140 / 1,177 (96.86%) |
| **75% — primary** | **0.748269** | **7,518 (75.18%)** | **2,482 (24.82%)** | **97.10%** | **218** | **959 / 1,177 (81.48%)** |
| 90% | 0.513672 | 8,888 (88.88%) | 1,112 (11.12%) | 93.41% | 586 | 591 / 1,177 (50.21%) |

At the primary cutoff, **7,300 test images were correctly accepted**, 218 were incorrectly accepted, and 2,482 were referred for review. The referred group contained 959 incorrect predictions and 1,523 correct predictions. The stricter cutoff reduced accepted errors to 37 but referred roughly half of all images; the more permissive cutoff accepted more images while leaving 586 errors in the accepted group. This quantifies the trade-off between review workload and mistakes that pass through automatically.

These are **routing results**, not measured outcomes after human review. Referring 959 incorrect predictions does not show that reviewers corrected them, and the 97.10% figure applies only to the accepted subset rather than the entire 10,000-image test set.

## Category-level limitation

A single cutoff did not produce uniform coverage or accuracy across actual categories. At the primary cutoff, 56.5% of frog images were referred, compared with 8.2% of horse images. Deer images had only 89.89% accuracy among accepted predictions, despite the aggregate accepted accuracy of 97.10%. Category-level checks therefore matter when deciding whether a global cutoff meets an operational requirement.

| Actual category | Accepted / 1,000 | Referred / 1,000 | Referral rate | Accuracy among accepted | Errors accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airplane | 784 | 216 | 21.6% | 96.68% | 26 |
| Automobile | 829 | 171 | 17.1% | 98.07% | 16 |
| Bird | 769 | 231 | 23.1% | 97.40% | 20 |
| Cat | 687 | 313 | 31.3% | 96.80% | 22 |
| Deer | 663 | 337 | 33.7% | 89.89% | 67 |
| Dog | 768 | 232 | 23.2% | 96.22% | 29 |
| Frog | 435 | 565 | 56.5% | 94.02% | 26 |
| Horse | 918 | 82 | 8.2% | 99.78% | 2 |
| Ship | 886 | 114 | 11.4% | 99.10% | 8 |
| Truck | 779 | 221 | 22.1% | 99.74% | 2 |

These groups use ground-truth categories for retrospective evaluation. Actual categories are unavailable when routing new unlabeled images. The disparities are differences across image categories; they do not establish a demographic fairness finding or explain the cause of the errors.

## Paired uncertainty

| Test accuracy comparison | Difference, percentage points | Paired bootstrap 95% interval, percentage points |
| --- | ---: | ---: |
| Photo prompt minus label only | +0.88 | +0.49 to +1.31 |
| Ensemble minus label only | +1.19 | +0.80 to +1.62 |
| Ensemble minus photo prompt | +0.31 | +0.05 to +0.59 |

The recorded intervals used 2,000 paired image bootstrap resamples with seed 2026. All three intervals were above zero in this run, while the ensemble's gain over the photo prompt was small. These descriptive intervals reuse the same test images for each method, are not adjusted for multiple comparisons, and do not quantify performance changes on a new business domain. They do not justify changing the development-selected method after inspecting test results.

## Business-practice discussion

The practical pattern is an image-tagging workflow that accepts some category suggestions automatically and queues lower-scoring suggestions for review. This experiment supplies the information a team would need to discuss that workflow: accepted volume, residual accepted errors, review volume, and uneven results across categories.

For example, the primary rule would create a queue of 2,482 reviews per 10,000 images **on this benchmark**, while leaving 218 mistakes among the accepted predictions. Whether that balance is useful depends on the actual task, the consequences of a wrong tag, reviewer capacity, and measured reviewer performance. The benchmark does not measure review time, successful corrections, operational costs, or financial returns.

Before applying the approach to a business image collection, define the relevant labels and acceptable error levels, evaluate on representative images, and set the prompt and threshold on separate development data. Then measure the complete workflow on held-out data, including category-level performance, reviewer decisions, and time to complete the queue. Real operational imagery, unfamiliar objects, and a different category mix may behave differently from CIFAR-10.


## Runtime and verification

The saved environment reports a Tesla T4, Python 3.13.16 and PyTorch 2.14.1+cu130. Encoding 10,000 test images took 55.67 seconds; scoring those shared vectors against all three prompt sets took 0.188 seconds. Model loading took 31.54 seconds and text prototype construction 0.798 seconds. These are single-run batched timings with setup recorded separately, not measured request latency or a production capacity estimate.

The archive verifier recomputed metrics from saved logits and checked all 36,000 prediction rows. Independent checks used scikit-learn for classification metrics and SciPy for softmax. Source checks matched saved IDs, labels and RGB hashes to the pinned CIFAR-10 files. Numerical tables and JSON outputs in the executed notebook matched the archive. Model inference was not repeated. File integrity and source alignment do not independently establish neural inference or execution chronology.

See `verification.json`, `independent_audit.json`, `source_alignment.json` and `package_changes.json` in this directory. Original result files are preserved unchanged under `results/`.

## Selected development errors

The original error worksheet is unchanged in the result directory. The separate [annotated worksheet](error_analysis.csv) records visual observations of its ten highest-score development errors. For example, row 35092 shows a small distant animal amid vertical background structures, row 9311 shows a dark wing-like silhouette against blue sky, and row 45583 has a bright subject whose body boundaries are difficult to resolve. Low resolution and ambiguous outlines are visible properties; this inspection does not establish what caused the model's prediction.

The ten examples include four deer-to-horse errors and two dog-to-cat errors. They are selected by confidence among errors, so those counts do not estimate the frequency of those confusions across the dataset. Benchmark labels were retained. Images were inspected from the pinned source and are not bundled in this repository.
