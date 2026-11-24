# Explaining the experiment

## Why compare these three prompts?

CLIP compares an image vector with a vector for each candidate description. “Airplane” and “a photo of an airplane” need not produce the same text vector. The experiment measures whether this difference changes predictions, instead of assuming descriptive prompts always help.

The ensemble describes each class five ways. Normalize each individual text vector, average the five vectors for that class, and normalize the average. Averaging unit vectors prevents text-vector magnitude from weighting a description more heavily. Renormalizing makes the final image–class dot product a cosine similarity. This fixed small ensemble is not the complete original CLIP ensemble.

## What determines a prediction?

`build_text_prototypes` returns three matrices, each with 10 rows and 512 columns. `encode_images` returns one normalized 512-value vector per image. `compute_logits` multiplies the image matrix by each class matrix's transpose and applies the checkpoint's learned positive logit scale. This yields ten scores per image. The largest score determines the predicted class; ties use the lowest class index.

Softmax turns the ten scores into nonnegative values summing to one. This is a relative distribution over the ten candidate labels. The maximum is used as a ranking score for abstention, not an established probability that the prediction is correct. Changing the labels or scale changes these scores.

## Why use separate development and threshold samples?

Prompt selection uses development labels to choose the highest-accuracy configuration. Threshold setting uses a different sample's maximum softmax scores. The test set supplies neither choice. `lock_protocol` does not accept test inputs, and its JSON is saved and hashed before test inference. This prevents an accidental code path from optimizing the operating point on the reported test outcomes; the export verifier cannot independently prove chronology.

`sample_splits` uses a fixed NumPy random generator and samples without replacement within each class. The saved row IDs identify exactly what was evaluated. Development and threshold IDs belong to the training split; official test IDs have their own namespace. Exact pixel hashes check image overlap across those namespaces.

## How does abstention work?

For the primary target, sort calibration scores conceptually from highest to lowest and use the score at rank `ceil(0.75 × N)` as the cutoff. The implementation uses NumPy partition to avoid sorting the entire vector. Accept scores `>= cutoff`; refer the rest for review. Ties may make calibration coverage higher than 75%, and test coverage can differ in either direction.

In the completed run, the primary threshold was 0.7482685294. It accepted 7,518 of 10,000 images, including 7,300 correct predictions and 218 errors. Coverage is 7,518/10,000 = 75.18%; selective accuracy is 7,300/7,518 = 97.10%. The 2,482 referred cases included 959 errors and 1,523 correct predictions. Referred errors cannot be counted as automatically corrected.

## How do the metrics differ?

Accuracy is the fraction of all predictions that match the benchmark labels. Macro F1 averages per-class F1 equally. Per-class recall asks how many actual examples of a class were recovered. The confusion matrix has actual classes in rows and predictions in columns.

Selective accuracy uses accepted examples only. `correctly_accepted_fraction_total` instead divides correctly accepted examples by all examples. Per-class coverage groups by the actual class and shows whether a class disproportionately contributes to review workload. Undefined ratios return `None` rather than inventing a zero or perfect score.

## Why use paired intervals?

Every method predicts the same test images. For a comparison, each image contributes +1 if only the second method is correct, −1 if only the first is correct, or 0 otherwise. Resampling the counts of these three outcomes with a multinomial is equivalent in distribution to resampling paired image indices. The 2.5th and 97.5th percentiles give the recorded descriptive interval. Pairing preserves that two methods can succeed or fail on the same images.

## What does verification establish?

The saved NPZ contains logits, labels, row IDs, and RGB hashes. The verifier recalculates softmax, predictions, metric denominators, prompt selection, thresholds, and paired intervals. It compares those with JSON and CSV outputs, checks split structure and text-vector norms, and validates the file manifest. It detects inconsistent exports and accidental file changes. It cannot prove that a particular real-world image deserves its label or independently rerun the model without its inputs and weights.

## What could change in a later study?

A later study could test different image domains, larger images, different class descriptions, calibrated scores, or a trained classifier. Those are new experimental choices. Once this test set has been inspected, tuning against its outcomes changes its role; use a new held-out evaluation for a new claim. A business pilot must measure task success and review cost with its own images and label definitions.

## How would I explain the trade-off in an interview?

The photo prompt won development selection by one correct image over the ensemble (891 versus 890). The ensemble subsequently scored 31 more correct test predictions, but switching the selected review method after seeing that result would change the evaluation protocol. I report both facts and retain the original selection.

The primary threshold concentrates accuracy in the accepted set while producing a review queue equal to 24.82% of the test set. A stricter threshold leaves fewer accepted errors but creates more work. Softmax can still be confidently wrong, and a global threshold treats categories unevenly: the accepted deer subset was only 89.89% accurate. A deployment decision needs representative business images and measured review/error costs; this experiment provides the evaluation pattern and the benchmark evidence.
