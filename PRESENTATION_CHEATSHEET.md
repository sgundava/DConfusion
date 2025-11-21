# DConfusion Presentation Cheat Sheet

## Quick Reference for Live Demo

### Installation & Launch
```bash
pip install dconfusion
streamlit run app/streamlit_app.py
# or
./run_app.sh
```

---

## Core API Patterns

### Creating Confusion Matrices
```python
from dconfusion import DConfusion

# Binary (positional args: TP, FN, FP, TN)
cm = DConfusion(85, 15, 10, 90)

# Multi-class
cm = DConfusion(
    confusion_matrix=[[50, 3, 2], [8, 45, 1], [4, 2, 48]],
    labels=['Cat', 'Dog', 'Bird']
)

# From predictions
cm = DConfusion.from_predictions(y_true, y_pred)

# From metrics (reverse engineering!)
cm = DConfusion.from_metrics(
    total_samples=100,
    accuracy=0.85,
    precision=0.80,
    recall=0.75
)
```

---

## 4 Standout Features

### 1. Warning System
```python
# Quick check
cm.print_warnings()

# Programmatic access
warnings = cm.check_warnings()
for w in warnings:
    print(w.severity, w.category, w.message)

# Compare two models
warnings = cm.compare_with(other_cm, metric='accuracy')
```

**Catches:** Small samples, imbalance, data leakage, misleading metrics

---

### 2. Statistical Testing

#### Bootstrap Confidence Intervals
```python
result = cm.get_bootstrap_confidence_interval(
    metric='accuracy',
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=42
)
# Returns: point_estimate, lower, upper, std_error
```

#### McNemar's Test
```python
result = model_a.mcnemar_test(model_b, alpha=0.05)
# Returns: statistic, p_value, significant, interpretation
```

**Key point:** "Are the models really different, or just noise?"

---

### 3. Cost-Sensitive Analysis

#### Calculate Business Cost
```python
# Total cost
total = cm.get_misclassification_cost(
    cost_fp=100,
    cost_fn=1000
)

# Average per sample
avg = cm.get_average_misclassification_cost(
    cost_fp=100,
    cost_fn=1000
)

# Full breakdown
summary = cm.get_cost_benefit_summary(
    cost_fp=100,
    cost_fn=1000,
    benefit_tp=50,
    benefit_tn=10
)
# Returns: total_cost, perfect_classifier_cost,
#          random_classifier_cost, savings, improvement
```

#### Find Optimal Metric
```python
rec = cm.find_optimal_metric_for_cost(
    cost_fp=1,
    cost_fn=10  # FN is 10x worse
)
# Returns: primary_recommendation ('recall', 'precision', etc),
#          explanation, cost_weighted_f_beta
```

**Use cases:**
- FN >> FP (ratio > 5): Optimize **recall** (medical, safety)
- FP >> FN (ratio < 0.2): Optimize **precision** (spam, marketing)
- Balanced (0.5-2): Optimize **F1** or **MCC**

#### Compare Models by Cost
```python
comparison = model_a.compare_cost_with(
    model_b,
    cost_fp=100,
    cost_fn=1000
)
# Returns: model1_total_cost, model2_total_cost,
#          cost_savings, better_model, recommendation
```

---

### 4. Metric Completion

#### Exact Reconstruction
```python
cm = DConfusion.from_metrics(
    total_samples=100,
    accuracy=0.85,
    precision=0.80,
    recall=0.75
)
# Returns: Complete DConfusion object
# Then: calculate any other metric!
```

**Requirements:** 3+ independent metrics + total_samples

#### Probabilistic Inference
```python
result = DConfusion.infer_metrics(
    total_samples=100,
    accuracy=0.85,
    prevalence=0.40,
    confidence_level=0.95,
    n_simulations=10000
)

# Access inferred metrics
precision = result['inferred_metrics']['precision']
# Returns: mean, median, ci_lower, ci_upper, std, min, max
```

**Use when:** Incomplete info, sensitivity analysis, paper validation

---

## Standard Metrics (Quick Reference)

```python
# Basic
cm.get_accuracy()
cm.get_precision()
cm.get_recall()        # aka sensitivity, TPR
cm.get_specificity()   # aka TNR
cm.get_f1_score()

# Advanced
cm.get_mcc()           # Matthews Correlation Coefficient (alias!)
cm.get_npv()           # Negative Predictive Value
cm.get_cohens_kappa()
cm.get_balanced_accuracy()

# Rate aliases (common abbreviations)
cm.get_tpr()           # True Positive Rate (same as recall)
cm.get_tnr()           # True Negative Rate (same as specificity)
cm.get_fpr()           # False Positive Rate
cm.get_fnr()           # False Negative Rate
cm.get_ppv()           # Positive Predictive Value (same as precision)

# All at once
metrics = cm.get_all_metrics()
```

---

## Key Talking Points

### What Makes DConfusion Unique?

1. **Research-backed warning system**
   - Based on Chicco, Lovell, Fazekas & Kovács
   - Catches data leakage, sample issues, misleading metrics

2. **Statistical rigor built-in**
   - Bootstrap CIs for any metric
   - McNemar's test for paired comparison
   - Uncertainty quantification

3. **Business-focused**
   - Cost-sensitive analysis
   - Metric recommendations based on cost ratios
   - ROI calculations

4. **Innovative metric completion**
   - Reconstruct confusion matrix from partial metrics
   - Validate published research
   - Infer missing metrics with confidence intervals

5. **Production-ready**
   - Clean mixin architecture
   - Binary + multi-class support
   - Web UI for stakeholders
   - CSV/JSON import/export

### Comparison to Alternatives

**vs scikit-learn:**
- sklearn gives you the matrix
- We give you the analysis (warnings, stats, costs, inference)

**vs manual calculation:**
- Catches mistakes you'd miss
- Research-backed recommendations
- Statistical testing built-in

**vs other tools:**
- Only tool with metric completion
- Only tool with research-based warnings
- Cost-sensitive analysis is unique

---

## Web UI Features

**6 Tabs in Model Comparison:**
1. Visualizations - Heatmaps side-by-side
2. Metrics Comparison - Sortable table + CSV export
3. Warnings & Quality - Automatic checks
4. Statistical Testing - Interactive CIs + McNemar
5. Cost Analysis - Custom cost structures
6. Detailed View - Individual matrix deep-dive

**Metric Completion Tab:**
- Reconstruct from metrics
- Infer with confidence intervals
- Interactive parameter adjustment

---

## Demo Flow (8-10 min)

1. **Problem** (1 min) - "Most tools miss critical issues"
2. **4 Features** (5-6 min)
   - Warnings (1.5 min)
   - Statistical Testing (1.5 min)
   - Cost Analysis (1.5 min)
   - Metric Completion (1.5 min) ← **Showstopper**
3. **Web UI** (1-2 min) - Quick tour
4. **Real-world example** (1 min) - Medical diagnosis
5. **Close** (30 sec) - Why DConfusion?

---

## Troubleshooting

### If demo fails:
- Have pre-computed outputs ready to show
- Screenshots of web UI
- Talk through the code without running

### If running slow:
- Reduce n_bootstrap to 100
- Reduce n_simulations to 1000
- Pre-compute expensive operations

### If questions during demo:
- "Great question - let me finish this section and come back to it"
- Keep momentum, don't derail the narrative

---

## Killer Sound Bites

- "DConfusion doesn't just calculate metrics - it helps you make better decisions"
- "Research-backed intelligence, not just calculation"
- "Caught data leakage twice in production this month"
- "The difference isn't statistically significant - most tools miss this"
- "Connect ML performance to ROI in 3 lines of code"
- "Reconstruct confusion matrices from research papers - I haven't seen this anywhere else"
- "All four basic rates, not just accuracy"
- "Stakeholders can explore models without writing code"

---

## Q&A Quick Answers

**"How does this compare to X?"**
→ "X does calculation. We do analysis. Warnings, statistics, costs, inference."

**"Multi-class support?"**
→ "Yes, fully supported throughout. Some advanced stats are binary-only currently."

**"Production ready?"**
→ "Yes. Semantic versioning, extensive validation, clean architecture, active maintenance."

**"Performance?"**
→ "Metrics are instant. Bootstrap takes seconds. Metric inference < 1s for 10k simulations."

**"Integration?"**
→ "Works with sklearn via from_predictions(), CSV/JSON I/O, easy to pipeline."

---

## Technical Backup Info

**Research References:**
- Chicco et al. - MCC advantages
- Lovell et al. - Uncertainty scales as 1/√N
- Fazekas & Kovács - Numerical consistency
- Efron & Tibshirani - Bootstrap methods
- McNemar (1947) - Paired comparison

**Architecture:**
```python
class DConfusion(
    DConfusionCore,
    MetricsMixin,
    VisualizationMixin,
    IOMixin,
    StatisticalTestsMixin,
    MetricInferenceMixin
)
```

**Metric Completion Math:**
- from_metrics: Solves system of equations analytically
- infer_metrics: Monte Carlo sampling of valid matrices

**Dependencies:**
- numpy, scipy, matplotlib (core)
- streamlit, pandas (web app)
- Python 3.8+

---

Good luck! 🎯