"""
Real-World Research Paper Scenarios

This demonstrates how the enhanced metric inference helps reproduce and validate
results from research papers that use different metric reporting conventions.
"""

from dconfusion import DConfusion

print("=" * 90)
print("RESEARCH PAPER REPRODUCTION SCENARIOS")
print("=" * 90)

# Scenario 1: Medical Diagnostic Test Paper
print("\n" + "=" * 90)
print("SCENARIO 1: Medical Diagnostic Test (Typical Medical Journal Format)")
print("=" * 90)

print("""
Hypothetical paper excerpt:
\"\"\"
We evaluated our COVID-19 rapid test on 500 patients. The test demonstrated:
- Sensitivity (TPR): 92%
- Specificity (TNR): 95%
- Disease prevalence in our sample: 18%
\"\"\"

Goal: Reconstruct the complete confusion matrix and calculate unreported metrics.
""")

medical_test = DConfusion.from_metrics(
    total_samples=500,
    tpr=0.92,  # Medical papers use TPR instead of recall
    tnr=0.95,  # Medical papers use TNR instead of specificity
    prevalence=0.18
)

print("Reconstructed Confusion Matrix:")
print(f"  TP: {medical_test.true_positive:3d}  |  FN: {medical_test.false_negative:3d}")
print(f"  FP: {medical_test.false_positive:3d}  |  TN: {medical_test.true_negative:3d}")

print("\nNow we can calculate the UNREPORTED metrics:")
print(f"  PPV (Positive Predictive Value):  {medical_test.get_precision():.3f}")
print(f"  NPV (Negative Predictive Value):  {medical_test.get_npv():.3f}")
print(f"  Accuracy:                          {medical_test.get_accuracy():.3f}")
print(f"  F1 Score:                          {medical_test.get_f1_score():.3f}")

print("\n💡 Insight: With 18% prevalence, even high sensitivity/specificity gives PPV of only 79%!")
print("   This means ~21% of positive tests are false alarms - important for clinical decisions!")

# Scenario 2: Machine Learning Paper with Type I/II Errors
print("\n" + "=" * 90)
print("SCENARIO 2: ML Paper Using Statistical Hypothesis Testing Terminology")
print("=" * 90)

print("""
Hypothetical paper excerpt:
\"\"\"
Our anomaly detection system was evaluated on 1000 network events:
- Type I Error (False Alarm Rate): 8%
- Type II Error (Miss Rate): 12%
- Base rate of anomalies: 5%
\"\"\"

Goal: Convert to ML metrics and assess practical performance.
""")

ml_system = DConfusion.from_metrics(
    total_samples=1000,
    fpr=0.08,  # Type I error
    fnr=0.12,  # Type II error
    prevalence=0.05
)

print("Reconstructed Confusion Matrix:")
print(f"  TP: {ml_system.true_positive:3d}  |  FN: {ml_system.false_negative:3d}")
print(f"  FP: {ml_system.false_positive:3d}  |  TN: {ml_system.true_negative:3d}")

print("\nTranslated to ML metrics:")
print(f"  Precision (PPV):    {ml_system.get_precision():.3f}")
print(f"  Recall (TPR):       {ml_system.get_recall():.3f}")
print(f"  Specificity (TNR):  {ml_system.get_specificity():.3f}")
print(f"  Accuracy:           {ml_system.get_accuracy():.3f}")
print(f"  F1 Score:           {ml_system.get_f1_score():.3f}")

print("\n💡 Insight: Despite 92% accuracy, precision is only 36%!")
print("   With low prevalence (5%), most alarms are false positives!")
print("   This system would overwhelm analysts with false alarms.")

# Scenario 3: Quality Control Paper
print("\n" + "=" * 90)
print("SCENARIO 3: Quality Control Study (Industrial Engineering Format)")
print("=" * 90)

print("""
Hypothetical paper excerpt:
\"\"\"
Our automated defect detection system inspected 2000 products:
- Error Rate: 6%
- Precision (Positive Predictive Value): 88%
- Defect rate (prevalence): 10%
\"\"\"

Goal: Determine if this meets quality standards (target: <5% false negatives).
""")

qc_system = DConfusion.from_metrics(
    total_samples=2000,
    error_rate=0.06,  # Instead of accuracy
    precision=0.88,
    prevalence=0.10
)

print("Reconstructed Confusion Matrix:")
print(f"  TP: {qc_system.true_positive:3d}  |  FN: {qc_system.false_negative:3d}")
print(f"  FP: {qc_system.false_positive:3d}  |  TN: {qc_system.true_negative:3d}")

print("\nQuality Metrics:")
print(f"  Recall (Detection Rate):           {qc_system.get_recall():.3f}")
print(f"  False Negative Rate (Miss Rate):   {qc_system.get_false_negative_rate():.3f}")
print(f"  Specificity (True Negative Rate):  {qc_system.get_specificity():.3f}")

fnr_percent = qc_system.get_false_negative_rate() * 100
if fnr_percent < 5:
    print(f"\n✓ PASS: False negative rate ({fnr_percent:.1f}%) meets <5% target!")
else:
    print(f"\n✗ FAIL: False negative rate ({fnr_percent:.1f}%) exceeds 5% target!")
    print(f"   {qc_system.false_negative} defective products would ship to customers!")

# Scenario 4: Comparative Study with Incomplete Reporting
print("\n" + "=" * 90)
print("SCENARIO 4: Comparing Two Models from Different Papers")
print("=" * 90)

print("""
Paper A reports (Medical format):
  - N=300, Sensitivity=0.90, Specificity=0.88, Prevalence=0.25

Paper B reports (ML format):
  - N=300, Precision=0.85, Recall=0.92, Prevalence=0.25

Goal: Which is better? Need complete metrics for fair comparison.
""")

model_a = DConfusion.from_metrics(
    total_samples=300,
    tpr=0.90,
    tnr=0.88,
    prevalence=0.25
)

model_b = DConfusion.from_metrics(
    total_samples=300,
    precision=0.85,
    recall=0.92,
    prevalence=0.25
)

print("Model A (from Paper A):")
print(f"  Precision: {model_a.get_precision():.3f}")
print(f"  Recall:    {model_a.get_recall():.3f}")
print(f"  F1 Score:  {model_a.get_f1_score():.3f}")
print(f"  Accuracy:  {model_a.get_accuracy():.3f}")

print("\nModel B (from Paper B):")
print(f"  Precision: {model_b.get_precision():.3f}")
print(f"  Recall:    {model_b.get_recall():.3f}")
print(f"  F1 Score:  {model_b.get_f1_score():.3f}")
print(f"  Accuracy:  {model_b.get_accuracy():.3f}")

# Statistical comparison
result = model_a.mcnemar_test(model_b)
print(f"\nMcNemar's Test:")
print(f"  P-value: {result['p_value']:.4f}")
print(f"  {result['interpretation']}")

if model_a.get_f1_score() > model_b.get_f1_score():
    winner = "Model A"
    f1_diff = (model_a.get_f1_score() - model_b.get_f1_score()) * 100
else:
    winner = "Model B"
    f1_diff = (model_b.get_f1_score() - model_a.get_f1_score()) * 100

print(f"\n💡 Conclusion: {winner} performs better (F1 difference: {f1_diff:.1f}%)")
if not result['significant']:
    print("   However, the difference is NOT statistically significant.")
    print("   More data would be needed for a definitive conclusion.")

# Scenario 5: Uncertainty Analysis for Small Sample
print("\n" + "=" * 90)
print("SCENARIO 5: Assessing Reliability of Small-Sample Study")
print("=" * 90)

print("""
Hypothetical paper excerpt:
\"\"\"
Pilot study with 80 samples showed:
- Accuracy: 87%
- Prevalence: 30%
\"\"\"

Goal: Quantify uncertainty in unreported metrics.
""")

result = DConfusion.infer_metrics(
    total_samples=80,
    accuracy=0.87,
    prevalence=0.30,
    confidence_level=0.95,
    n_simulations=10000,
    random_state=42
)

print(f"Given only accuracy and prevalence, possible ranges for other metrics:")
print(f"\n{'Metric':<15} {'Mean':>8} {'95% CI':>25}")
print("-" * 50)

for metric in ['precision', 'recall', 'specificity', 'f1_score', 'npv']:
    stats = result['inferred_metrics'][metric]
    ci_width = stats['ci_upper'] - stats['ci_lower']
    print(f"{metric:<15} {stats['mean']:>8.3f}  [{stats['ci_lower']:.3f} - {stats['ci_upper']:.3f}]  (±{ci_width:.3f})")

print("\n💡 Insight: With only 80 samples and limited information:")
print("   - Precision could be anywhere from 0.25 to 1.00!")
print("   - Need more samples OR more reported metrics for reliable inference")
print("   - This quantifies the uncertainty ignored by papers that don't report CIs")

# Summary
print("\n" + "=" * 90)
print("SUMMARY: VALUE OF ENHANCED METRIC INFERENCE")
print("=" * 90)
print("""
✓ Reproduce results from papers with different reporting conventions
✓ Calculate unreported metrics for complete picture
✓ Compare models across papers with inconsistent metric reporting
✓ Assess practical implications (e.g., false alarm rates)
✓ Quantify uncertainty when information is incomplete
✓ Validate reported metrics for consistency
✓ Bridge medical/statistical/ML terminology differences

This makes DConfusion essential for:
- Literature reviews and meta-analyses
- Reproducing published results
- Comparing models from different sources
- Assessing real-world implications
- Research validation and peer review
""")

print("=" * 90)
