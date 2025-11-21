# DConfusion Implementation Plan

## Priority 1: Method Aliases (30 minutes)

### Task: Add common abbreviations as method aliases

**File:** `dconfusion/metrics.py`

**Add these methods after existing metric methods:**

```python
# Aliases for common abbreviations
def get_mcc(self) -> float:
    """Alias for get_matthews_correlation_coefficient()."""
    return self.get_matthews_correlation_coefficient()

def get_npv(self) -> float:
    """
    Calculate Negative Predictive Value (NPV).

    NPV = TN / (TN + FN)

    Represents the probability that a negative prediction is correct.
    """
    if self.n_classes != 2:
        raise ValueError("NPV is only available for binary classification")

    denominator = self.true_negative + self.false_negative
    if denominator == 0:
        raise ZeroDivisionError("Cannot calculate NPV: no actual negatives")

    return self.true_negative / denominator

def get_tpr(self) -> float:
    """Alias for get_recall() - True Positive Rate / Sensitivity."""
    return self.get_recall()

def get_tnr(self) -> float:
    """Alias for get_specificity() - True Negative Rate."""
    return self.get_specificity()

def get_fpr(self) -> float:
    """Alias for get_false_positive_rate()."""
    return self.get_false_positive_rate()

def get_fnr(self) -> float:
    """Alias for get_false_negative_rate()."""
    return self.get_false_negative_rate()

def get_ppv(self) -> float:
    """Alias for get_precision() - Positive Predictive Value."""
    return self.get_precision()
```

**Testing:**
```python
# Quick test
cm = DConfusion(85, 15, 10, 90)
assert cm.get_mcc() == cm.get_matthews_correlation_coefficient()
assert cm.get_tpr() == cm.get_recall()
assert cm.get_ppv() == cm.get_precision()
print("✓ All aliases work!")
```

**Impact:** Cleaner demo code, better UX, matches academic conventions

---

## Priority 2: Fix `from_metrics()` Precision (2-4 hours)

### Task: Improve reconstruction accuracy to < 0.5% error

**Problem:** Current implementation uses greedy rounding causing ~1-2% error

**Current approach (lines 521-548 in statistics.py):**
```python
# Loops through TP values, uses rounding
for tp in range(N + 1):
    actual_positives = round(tp / recall)  # ROUNDING ERROR
    predicted_positives = round(tp / precision)  # MORE ROUNDING
```

**Solution 1: Use optimization as primary method**

Replace Approach 3 with direct call to `_optimize_confusion_matrix`:

```python
# In _solve_confusion_matrix()
# Approach 3: Precision, Recall, Accuracy
if precision is not None and recall is not None and accuracy is not None:
    # Use optimization instead of greedy search
    result = cls._optimize_confusion_matrix(
        N, accuracy, precision, recall, None, None, None
    )
    if result is not None:
        # Verify solution quality
        tp, fn, fp, tn = result
        calc_acc = (tp + tn) / N
        calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Check if within tight tolerance
        if (abs(calc_acc - accuracy) < 0.005 and
            abs(calc_prec - precision) < 0.005 and
            abs(calc_rec - recall) < 0.005):
            return result
```

**Solution 2: Add validation and retry**

After reconstruction, validate and raise error if too far off:

```python
# After line 462 in from_metrics()
# Validate reconstruction quality
validation = reconstructed.check_metric_consistency(
    provided_metrics,
    tolerance=0.01  # 1% tolerance
)

if not validation['consistent']:
    # Try optimization approach
    from scipy.optimize import differential_evolution
    # ... more sophisticated optimization
```

**Testing:**
```python
# Test case from demo
cm = DConfusion.from_metrics(
    total_samples=100,
    accuracy=0.85,
    precision=0.80,
    recall=0.75
)

# Should now be within 0.5%
assert abs(cm.get_accuracy() - 0.85) < 0.005
assert abs(cm.get_precision() - 0.80) < 0.005
assert abs(cm.get_recall() - 0.75) < 0.005
```

**Impact:** Demo 6 shows exact/near-exact reconstruction, validates the feature

---

## Priority 3: ROC/PR Curve Plotting (4-6 hours)

### Task: Add threshold-based curve visualization

**Challenge:** Current design doesn't store predictions/probabilities, only final matrix

**Solution 1: Add probability-aware factory method**

```python
# In io.py
@classmethod
def from_predictions_proba(cls, y_true, y_proba, threshold=0.5, name=None):
    """
    Create confusion matrix from predictions with probabilities.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (0-1)
        threshold: Decision threshold
        name: Optional model name

    Returns:
        DConfusion object with stored probabilities for ROC/PR curves
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = cls.from_predictions(y_true, y_pred)

    # Store additional data for curves
    cm._y_true = y_true
    cm._y_proba = y_proba
    cm._has_proba = True

    return cm
```

**Solution 2: Add plotting methods**

```python
# In visualization.py
def plot_roc_curve(self, ax=None, **kwargs):
    """
    Plot ROC curve if probabilities are available.

    Raises:
        ValueError: If created without probabilities
    """
    if not hasattr(self, '_has_proba') or not self._has_proba:
        raise ValueError(
            "ROC curve requires probability scores. "
            "Use DConfusion.from_predictions_proba() instead."
        )

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(self._y_true, self._y_proba)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', **kwargs)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax.figure if ax.figure else fig

def plot_precision_recall_curve(self, ax=None, **kwargs):
    """Plot Precision-Recall curve."""
    # Similar implementation
```

**Usage:**
```python
# Create with probabilities
cm = DConfusion.from_predictions_proba(y_true, y_proba)

# Plot curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cm.plot_roc_curve(ax=ax1)
cm.plot_precision_recall_curve(ax=ax2)
plt.show()
```

**Impact:** Major visualization upgrade, useful for model selection

---

## Priority 4: Threshold Optimization (6-8 hours)

### Task: Find optimal decision threshold

**Implementation:**

```python
# In statistics.py or new module optimization.py

def find_optimal_threshold(
    y_true,
    y_proba,
    metric='f1_score',
    constraint=None,
    n_thresholds=100
):
    """
    Find decision threshold that optimizes a metric.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1_score', 'accuracy', 'mcc', etc.)
        constraint: Optional constraint dict, e.g., {'fpr': 0.05, 'recall': 0.90}
        n_thresholds: Number of thresholds to try

    Returns:
        dict with:
            - optimal_threshold: Best threshold
            - optimal_value: Metric value at optimal threshold
            - confusion_matrix: DConfusion at optimal threshold
            - threshold_curve: All thresholds and metric values
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    metric_values = []
    valid_thresholds = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = DConfusion.from_predictions(y_true, y_pred)

        # Check constraints
        if constraint:
            satisfies = True
            for constraint_metric, constraint_value in constraint.items():
                actual = getattr(cm, f'get_{constraint_metric}')()
                if constraint_metric in ['fpr', 'fnr']:
                    # Lower is better
                    if actual > constraint_value:
                        satisfies = False
                else:
                    # Higher is better
                    if actual < constraint_value:
                        satisfies = False

            if not satisfies:
                continue

        # Calculate metric
        try:
            value = getattr(cm, f'get_{metric}')()
            metric_values.append(value)
            valid_thresholds.append(threshold)
        except:
            continue

    if not valid_thresholds:
        raise ValueError("No threshold satisfies the constraints")

    # Find optimal
    optimal_idx = np.argmax(metric_values)
    optimal_threshold = valid_thresholds[optimal_idx]
    optimal_value = metric_values[optimal_idx]

    # Create confusion matrix at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    optimal_cm = DConfusion.from_predictions(y_true, y_pred_optimal)

    return {
        'optimal_threshold': optimal_threshold,
        'optimal_value': optimal_value,
        'confusion_matrix': optimal_cm,
        'all_thresholds': valid_thresholds,
        'all_values': metric_values
    }
```

**Visualization helper:**

```python
def plot_threshold_curve(result, metric_name='F1 Score'):
    """Plot metric value across thresholds."""
    plt.figure(figsize=(10, 6))
    plt.plot(result['all_thresholds'], result['all_values'])
    plt.axvline(result['optimal_threshold'], color='r', linestyle='--',
                label=f'Optimal = {result["optimal_threshold"]:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

**Usage:**
```python
# Find threshold that maximizes F1
result = find_optimal_threshold(y_true, y_proba, metric='f1_score')
print(f"Optimal threshold: {result['optimal_threshold']:.3f}")
print(f"F1 Score: {result['optimal_value']:.3f}")

# Find threshold that keeps FPR < 5% while maximizing recall
result = find_optimal_threshold(
    y_true, y_proba,
    metric='recall',
    constraint={'fpr': 0.05}
)
```

**Integration with cost-sensitive:**
```python
# Find threshold that minimizes total cost
def find_cost_optimal_threshold(y_true, y_proba, cost_fp, cost_fn):
    """Find threshold that minimizes misclassification cost."""
    thresholds = np.linspace(0, 1, 100)
    costs = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = DConfusion.from_predictions(y_true, y_pred)
        cost = cm.get_misclassification_cost(cost_fp, cost_fn)
        costs.append(cost)

    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx]
```

**Impact:** Practical tool for deployment, complements cost-sensitive analysis

---

## Priority 5: Multi-class Metric Completion (20-30 hours)

### Task: Extend from_metrics() to multi-class

**Challenge:** Much more complex - multi-class has N×N matrix (many more unknowns)

**Scope:**
- Need N² values (for N classes)
- Only have ~N metrics typically (accuracy + per-class precision/recall)
- Underconstrained problem in general

**Approach:**

```python
@classmethod
def from_multiclass_metrics(
    cls,
    num_classes: int,
    total_samples: int,
    accuracy: float,
    per_class_precision: Dict[str, float],
    per_class_recall: Dict[str, float],
    class_distribution: Optional[Dict[str, float]] = None
):
    """
    Reconstruct multi-class confusion matrix from metrics.

    This is more complex than binary case and may have multiple solutions.
    Uses constrained optimization to find one valid solution.

    Args:
        num_classes: Number of classes
        total_samples: Total number of samples
        accuracy: Overall accuracy
        per_class_precision: Precision for each class
        per_class_recall: Recall for each class
        class_distribution: Optional true class distribution

    Returns:
        DConfusion object (multi-class)

    Note:
        - Multi-class reconstruction is underconstrained
        - May not have unique solution
        - Returns one valid solution if multiple exist
    """
    # This is a research-level problem
    # Would need sophisticated optimization
    pass
```

**Why save for last:**
- Most complex mathematically
- Research-level problem (might need paper review)
- Lower immediate impact (binary is most common)
- Could be PhD thesis topic on its own

**Alternative:** Start with simpler case
```python
def from_multiclass_metrics_simple(
    cls,
    num_classes: int,
    total_samples: int,
    accuracy: float,
    assume_symmetric_confusion=True
):
    """
    Simplified multi-class reconstruction assuming classes are confused equally.
    Much easier problem but less general.
    """
```

---

## Summary: Recommended Order

1. ✅ **Method Aliases** (30 min) - Do this TODAY before presentation
2. ✅ **Fix from_metrics()** (2-4 hours) - Do this THIS WEEK
3. ✅ **ROC/PR Curves** (4-6 hours) - High visibility feature
4. ✅ **Threshold Optimization** (6-8 hours) - Practical and cool
5. ✅ **Multi-class Metric Completion** (20-30 hours) - Long-term project

**Time to complete 1-4:** ~2 weeks of focused work
**Time for all 5:** 1-2 months

## For Tomorrow's Presentation

When discussing future work, mention:
1. "We're adding method aliases for better UX - `cm.get_mcc()` instead of the long form"
2. "Improving from_metrics() precision to sub-0.5% error"
3. "ROC and PR curve visualization for threshold analysis"
4. "Threshold optimization integrated with cost-sensitive analysis"
5. "Long-term: extending metric completion to multi-class (research problem)"

This shows you have a concrete roadmap with quick wins AND ambitious long-term goals.
