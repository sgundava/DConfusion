# DConfusion Warning System Guide

## Overview

The DConfusion warning system helps identify potential pitfalls in confusion matrix analysis based on peer-reviewed research on binary classification metrics. This system was developed based on findings from:

1. **Chicco et al.** - Studies on MCC advantages and ROC AUC limitations in binary classification
2. **Lovell et al.** - Research on uncertainty in classification metrics and sample size requirements
3. **Fazekas & Kovács** - Work on numerical consistency testing in ML performance evaluation

## Key Concepts

### The Replication Crisis

Research has shown that a significant portion of published ML results cannot be replicated:
- Only 6% of 208 defect prediction studies were successfully replicated
- ~30% of medical imaging papers contain inconsistent performance scores
- Common causes: data leakage, undisclosed methodologies, insufficient sample sizes

### Uncertainty Dominates Small Samples

**Critical insight from Lovell et al.:** Uncertainty in ALL metrics scales as 1/√N

- Need **4x more data** to halve uncertainty
- Small differences between models may be meaningless if uncertainty distributions overlap
- No metric fixes the fundamental problem of insufficient data

### The Problem with Class Imbalance

The real issue isn't the balance ratio itself, but the **absolute number of samples per class**:
- 10/90 split with 1000 samples (100/900) is better than 50/50 with 20 samples (10/10)
- Both classes need sufficient samples for reliable metrics

## Using the Warning System

### Basic Usage

```python
from dconfusion import DConfusion

# Create a confusion matrix
cm = DConfusion(true_positive=10, false_negative=5,
                false_positive=3, true_negative=12)

# Print formatted warnings
cm.print_warnings()

# Or access warnings programmatically
warnings = cm.check_warnings()
for warning in warnings:
    print(warning)
```

### Comparing Models

```python
model_a = DConfusion(tp=48, fn=7, fp=5, tn=40)
model_b = DConfusion(tp=50, fn=5, fp=8, tn=37)

# Compare with automatic warning checks
result = model_a.compare_with(model_b, metric='accuracy')

if result['has_warnings']:
    for warning in result['warnings']:
        print(warning)
```

### Filtering by Severity

```python
# Get only critical warnings
from dconfusion import WarningSeverity

warnings = cm.check_warnings()
critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]

# Print without INFO level warnings
cm.print_warnings(include_info=False)
```

## Warning Categories

### 1. Sample Size Warnings

**What it checks:** Whether you have enough total samples for reliable metrics

**Threshold:** 100 samples recommended minimum

**Why it matters:** Uncertainty scales as 1/√N. With 25 samples instead of 100, uncertainty is 2x higher.

**Example:**
```
[WARNING] Sample Size: Total sample size (20) is small.
Metric uncertainty is approximately 2.2x higher than with 100 samples.

Recommendation: Collect 80 more samples to reduce uncertainty,
or report confidence intervals with all metrics.
```

**What to do:**
- Collect more data if possible
- Report confidence intervals
- Be cautious about small differences between models
- Consider whether your sample is representative

### 2. Class Imbalance Warnings

**What it checks:**
- Absolute number of samples in each class
- Severe imbalance ratios (< 1% minority class)

**Threshold:** 30 samples per class recommended minimum

**Why it matters:** Metrics for classes with few samples have high uncertainty, regardless of overall accuracy

**Example:**
```
[WARNING] Class Imbalance: Minority class (positive) has only 8 samples.
Metrics for this class will have high uncertainty.

Recommendation: Collect at least 30 samples of the positive class
for reliable metric estimates.
```

```
[CRITICAL] Severe Class Imbalance: Extreme class imbalance detected:
minority class represents only 0.80% of samples (8/1000).

Recommendation: Consider: (1) collecting more minority class samples,
(2) using stratified sampling, or (3) being especially cautious when
interpreting metrics like precision and F1 score.
```

**What to do:**
- Collect more samples of minority class
- Use stratified sampling
- Consider oversampling/undersampling techniques
- Use appropriate metrics (MCC, balanced accuracy)
- Be extra cautious with precision and recall

### 3. Empty Cell Warnings

**What it checks:** Zero values in TP, TN, FP, or FN

**Why it matters:**
- Zero TP means the model failed to identify ANY positive cases
- Zero TN means the model failed to identify ANY negative cases
- These make some metrics undefined or zero

**Examples:**
```
[CRITICAL] Zero True Positives: No true positives (TP=0).
Precision and recall are undefined or zero.

Recommendation: The model failed to correctly identify any positive cases.
Check if the model is trained correctly or if the threshold is appropriate.
```

```
[INFO] Zero False Positives: No false positives (FP=0).
While this seems good, verify that the model isn't simply predicting all negatives.

Recommendation: Check the distribution of predictions and consider
if the model is too conservative.
```

**What to do:**
- Check model training process
- Adjust decision threshold
- Verify prediction distribution
- Check for data quality issues

### 4. High Metric Uncertainty

**What it checks:** Whether metric confidence intervals are wide

**Threshold:** Relative uncertainty > 10%

**Why it matters:** Large uncertainty means small differences may not be meaningful

**Example:**
```
[WARNING] High Metric Uncertainty: Metric uncertainty is high
(±10.7% for accuracy). With n=20, differences smaller than ~21.3%
may not be meaningful.

Recommendation: Need 80 samples to halve the uncertainty, or always
report confidence intervals when comparing models.
```

**What to do:**
- Report confidence intervals
- Don't over-interpret small differences
- Collect more data
- Use statistical tests for comparisons

### 5. Perfect Classification Warning

**What it checks:** Whether accuracy = 100%

**Why it matters:** Perfect results often indicate methodological issues

**Example:**
```
[WARNING] Perfect Classification: Model achieved perfect classification
(100% accuracy). This is rare in practice.

Recommendation: Verify: (1) No data leakage between train/test sets,
(2) Proper cross-validation, (3) Target variable not included as feature,
(4) Test set is representative of real data. Perfect results often
indicate methodological issues.
```

**What to do:**
- Check for data leakage
- Verify train/test split
- Ensure target not in features
- Check if test set is too easy/not representative
- Verify cross-validation methodology

### 6. Poor Basic Rates Warning

**What it checks:** Whether all four basic rates (TPR, TNR, PPV, NPV) are adequate

**Threshold:** < 0.7 for any rate

**Why it matters:** High accuracy or ROC AUC can hide poor performance in specific rates

**Example:**
```
[WARNING] Poor Basic Rates: One or more basic rates are below 0.7:
Sensitivity/Recall (TPR=0.250), Precision (PPV=0.500).

Recommendation: High accuracy or ROC AUC can hide poor performance in
specific rates. Consider whether these poor rates are acceptable for
your application, or if model improvement or threshold adjustment is needed.
```

**What to do:**
- Always report all four basic rates
- Consider application-specific costs of different error types
- Adjust threshold if appropriate
- Improve model if rates are unacceptable

### 7. Misleading Accuracy Warning

**What it checks:** Whether high accuracy might be due to predicting majority class

**Why it matters:** Accuracy close to majority class proportion suggests trivial model

**Example:**
```
[WARNING] Potentially Misleading Accuracy: Accuracy (0.900) is close
to the majority class proportion (0.900). The model may not be learning
meaningful patterns.

Recommendation: Check if the model is simply predicting the majority class.
Consider using balanced metrics like MCC, F1 score, or balanced accuracy.
Always report all four basic rates (TPR, TNR, PPV, NPV).
```

**What to do:**
- Check prediction distribution
- Use balanced metrics (MCC, balanced accuracy)
- Report all basic rates
- Consider if model is actually learning

### 8. Metric Selection Guidance (INFO)

**What it provides:** General guidance on metric interpretation

**Example:**
```
[INFO] Metric Selection Guidance: Different metrics emphasize different
aspects of classifier performance. No single metric is 'best' for all contexts.

Recommendation: Report multiple metrics: (1) All four basic rates
(Sensitivity, Specificity, Precision, NPV), (2) A summary metric appropriate
for your use case (MCC for balanced view, F1 for precision-recall tradeoff,
etc.), (3) Confidence intervals or uncertainty estimates when comparing models.
```

### 9. Comparison Warnings

**What it checks when comparing two models:**
- Different sample sizes
- Small sample sizes
- Whether differences are meaningful given uncertainty

**Examples:**
```
[WARNING] Comparison: Different Sample Sizes: Models evaluated on
different sample sizes (100 vs 150). Direct comparison may be misleading.

Recommendation: Ideally, compare models on the same test set. If not
possible, use confidence intervals to account for different uncertainty levels.
```

```
[WARNING] Comparison: Small Difference: Accuracy difference (0.0100)
is smaller than ~2 standard errors (0.1000). This difference may not
be statistically significant.

Recommendation: Perform a proper statistical test (e.g., McNemar's test
for paired data, or proportion z-test) or calculate confidence intervals
to assess significance.
```

## Best Practices

### When Reporting Results

1. **Always report comprehensively:**
   - All four basic rates (TPR, TNR, PPV, NPV)
   - Chosen summary metric (MCC, F1, accuracy, etc.)
   - Uncertainty estimates or confidence intervals
   - Sample size and class distribution

2. **Check warnings before publishing:**
   ```python
   cm.print_warnings()
   if any(w.severity == WarningSeverity.CRITICAL for w in cm.check_warnings()):
       print("⚠ Critical issues detected! Review before publishing.")
   ```

3. **Document methodology:**
   - Describe evaluation procedure
   - Report cross-validation strategy
   - Share code and data if possible

### When Comparing Models

1. **Use same test set** when possible

2. **Check if differences are meaningful:**
   ```python
   result = model1.compare_with(model2)
   if result['has_warnings']:
       print("Comparison may not be reliable!")
   ```

3. **Use statistical tests:**
   - McNemar's test for paired predictions
   - Confidence interval overlap
   - Permutation tests

4. **Consider practical significance** vs statistical significance

### When Sample Size is Limited

1. **Report uncertainty:**
   ```python
   ci = cm.get_metric_confidence_interval('accuracy', confidence=0.95)
   print(f"Accuracy: {cm.accuracy:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})")
   ```

2. **Be conservative in conclusions**

3. **Focus on effect size** not just p-values

4. **Consider collecting more data** before making strong claims

## Research References

The warning system is based on:

1. **Chicco, D., & Jurman, G.** (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." *BMC Genomics*.

2. **Chicco, D., et al.** (2021). "The Matthews correlation coefficient (MCC) should replace the ROC AUC as the standard metric for assessing binary classification." *BioData Mining*.

3. **Lovell, D., et al.** (2022). "Asymptotically minimax uncertainty on classification errors." *Pattern Recognition*.

4. **Fazekas, G., & Kovács, L.** (2024). "mlscorecheck: Testing the Consistency of Performance Scores Reported for Machine Learning Models." *Software Impacts*.

## API Reference

### Classes

#### `ConfusionMatrixWarning`
Represents a single warning.

**Attributes:**
- `severity`: WarningSeverity enum (CRITICAL, WARNING, INFO)
- `category`: String category (e.g., "Sample Size")
- `message`: Description of the issue
- `recommendation`: Suggested action (optional)

#### `WarningSeverity`
Enum with three levels:
- `CRITICAL`: Severe issues that likely invalidate results
- `WARNING`: Issues that may affect interpretation
- `INFO`: General guidance and best practices

#### `WarningChecker`
Checks confusion matrices for issues.

**Methods:**
- `check_all()`: Run all checks, return list of warnings
- `get_warnings_by_severity(severity)`: Filter by severity
- `has_critical_warnings()`: Boolean check
- `format_warnings(include_info=True)`: Formatted string

### Functions

#### `check_comparison_validity(matrix1, matrix2)`
Check if comparing two confusion matrices is meaningful.

**Returns:** List of warnings about the comparison

**Example:**
```python
from dconfusion import check_comparison_validity

warnings = check_comparison_validity(cm1, cm2)
for w in warnings:
    print(w)
```

### DConfusion Methods

#### `check_warnings(include_info=True)`
Check for warnings, return list.

#### `print_warnings(include_info=True)`
Print formatted warning report.

#### `compare_with(other, metric='accuracy', show_warnings=True)`
Compare with another confusion matrix.

**Returns:** Dictionary with:
- `metric`: Name of compared metric
- `value1`, `value2`: Metric values
- `difference`: value1 - value2
- `relative_difference`: Percentage difference
- `better_model`: Which model is better
- `warnings`: List of comparison warnings (if `show_warnings=True`)
- `has_warnings`: Boolean flag

## Examples

See `examples/warnings_demo.py` for comprehensive examples of:
- Small sample size detection
- Class imbalance warnings
- Perfect classification detection
- Zero cell warnings
- Misleading accuracy detection
- Model comparison with warnings
- Programmatic access to warnings

## FAQ

**Q: Should I always aim for zero warnings?**

A: Not necessarily. INFO warnings are just guidance. Some warnings reflect constraints of your data (e.g., sample size) that you can't immediately change. The key is to be aware of limitations and report them.

**Q: What if I can't get more data?**

A: Focus on:
- Reporting confidence intervals
- Being conservative in conclusions
- Using multiple metrics
- Clear documentation of limitations

**Q: Are these thresholds absolute?**

A: No, they're research-informed guidelines. Context matters. A medical diagnostic tool may need higher standards than a content recommendation system.

**Q: Which metrics should I report?**

A: At minimum:
- All four basic rates (TPR, TNR, PPV, NPV)
- A summary metric appropriate to your context
- Confidence intervals when comparing models

**Q: How do I choose between MCC, F1, and accuracy?**

A:
- **MCC**: Balanced summary of all four rates, good default
- **F1**: When precision and recall matter more than specificity
- **Accuracy**: When classes are balanced and all errors equal
- **Context**: Consider costs of FP vs FN in your application

**Q: Can I customize warning thresholds?**

A: Currently no, but this may be added in future versions. For now, you can access warnings programmatically and apply your own filters.

## Contributing

Found an issue or have suggestions for additional warnings? Please open an issue or pull request on the GitHub repository.

## License

This warning system is part of DConfusion and uses the same license as the main package.
