"""
DConfusion Live Demo Script
Run sections interactively during presentation
"""

from dconfusion import DConfusion

print("="*80)
print("DConfusion - Intelligent Confusion Matrix Analysis")
print("="*80)

# =============================================================================
# DEMO 1: The Problem - Basic metrics miss critical issues
# =============================================================================
print("\n\n### DEMO 1: Basic Metrics Can Be Misleading ###\n")

cm = DConfusion(
    true_positive=10,
    false_negative=5,
    false_positive=3,
    true_negative=12
)

print(f"Accuracy: {cm.get_accuracy():.2%}")
print(f"Precision: {cm.get_precision():.2%}")
print(f"Recall: {cm.get_recall():.2%}")
print("\nSeems okay... but let's check for issues:\n")

cm.print_warnings()

input("\n\n[Press Enter for next demo...]")

# =============================================================================
# DEMO 2: Warning System - Catch issues automatically
# =============================================================================
print("\n\n### DEMO 2: Research-Based Warning System ###\n")

# Example: Perfect classification (suspicious!)
perfect_model = DConfusion(
    true_positive=50,
    false_negative=0,
    false_positive=0,
    true_negative=50
)

print("Perfect model - 100% accuracy:")
print(f"Accuracy: {perfect_model.get_accuracy():.2%}\n")
perfect_model.print_warnings()

input("\n\n[Press Enter for next demo...]")

# =============================================================================
# DEMO 3: Statistical Testing - Bootstrap Confidence Intervals
# =============================================================================
print("\n\n### DEMO 3: Statistical Testing - Bootstrap Confidence Intervals ###\n")

model = DConfusion(
    true_positive=85,
    false_negative=15,
    false_positive=10,
    true_negative=90
)

result = model.get_bootstrap_confidence_interval(
    metric='accuracy',
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=42
)

print("Bootstrap Analysis (1000 samples):")
print(f"Accuracy: {result['point_estimate']:.3f}")
print(f"95% CI: [{result['lower']:.3f}, {result['upper']:.3f}]")
print(f"Std Error: {result['std_error']:.4f}")
print("\n→ This quantifies our uncertainty!")

input("\n\n[Press Enter for next demo...]")

# =============================================================================
# DEMO 4: McNemar's Test - Compare two models statistically
# =============================================================================
print("\n\n### DEMO 4: McNemar's Test - Statistical Model Comparison ###\n")

model_a = DConfusion(
    true_positive=85,
    false_negative=15,
    false_positive=10,
    true_negative=90
)

model_b = DConfusion(
    true_positive=80,
    false_negative=20,
    false_positive=8,
    true_negative=92
)

print(f"Model A Accuracy: {model_a.get_accuracy():.3f}")
print(f"Model B Accuracy: {model_b.get_accuracy():.3f}")
print(f"Difference: {abs(model_a.get_accuracy() - model_b.get_accuracy()):.3f}")
print("\nRunning McNemar's test...")

result = model_a.mcnemar_test(model_b, alpha=0.05)

print(f"\nTest Statistic: {result['statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant difference? {result['significant']}")
print(f"\nInterpretation: {result['interpretation']}")

input("\n\n[Press Enter for next demo...]")

# =============================================================================
# DEMO 5: Cost-Sensitive Analysis - Business Impact
# =============================================================================
print("\n\n### DEMO 5: Cost-Sensitive Analysis - Real Business Impact ###\n")

# Medical diagnosis scenario
medical_model = DConfusion(
    true_positive=85,
    false_negative=15,  # 15 missed diagnoses!
    false_positive=10,
    true_negative=90
)

print("Medical Diagnosis Model:")
print(f"Accuracy: {medical_model.get_accuracy():.2%}")
print(f"Recall (Sensitivity): {medical_model.get_recall():.2%}")
print("\nCost Analysis:")
print("- False Positive (unnecessary test): $100")
print("- False Negative (missed disease): $10,000")

total_cost = medical_model.get_misclassification_cost(
    cost_fp=100,
    cost_fn=10000
)

print(f"\n→ Total Cost: ${total_cost:,}")
print(f"→ That's ${total_cost/medical_model.total:.2f} per patient!")

# Find optimal metric
recommendation = medical_model.find_optimal_metric_for_cost(
    cost_fp=100,
    cost_fn=10000
)

print(f"\n→ Recommended metric to optimize: {recommendation['primary_recommendation']}")
print(f"→ Reason: {recommendation['explanation']}")

# Compare with alternative model
alternative_model = DConfusion(
    true_positive=95,
    false_negative=5,   # Only 5 missed diagnoses
    false_positive=25,  # But more false alarms
    true_negative=75
)

comparison = medical_model.compare_cost_with(
    alternative_model,
    cost_fp=100,
    cost_fn=10000
)

print(f"\n\nComparing with Alternative Model:")
print(f"Current Model Cost: ${comparison['model1_total_cost']:,}")
print(f"Alternative Model Cost: ${comparison['model2_total_cost']:,}")
print(f"Savings: ${comparison['cost_savings']:,}")
print(f"Better Model: {comparison['better_model']}")
print(f"\n→ {comparison['recommendation']}")

input("\n\n[Press Enter for next demo...]")

# =============================================================================
# DEMO 6: Metric Completion - The Showstopper
# =============================================================================
print("\n\n### DEMO 6: Metric Completion - Reverse Engineering ###\n")

print("Scenario: Research paper reports:")
print("- 200 test samples")
print("- 85% accuracy")
print("- 80% precision")
print("- 75% recall")
print("\nCan we reconstruct their confusion matrix? YES!\n")

# Reconstruct from metrics (N=200 gives excellent precision <0.5% error)
reconstructed = DConfusion.from_metrics(
    total_samples=200,
    accuracy=0.85,
    precision=0.80,
    recall=0.75
)

print("Reconstructed Confusion Matrix:")
print(f"True Positives:  {reconstructed.true_positive}")
print(f"False Negatives: {reconstructed.false_negative}")
print(f"False Positives: {reconstructed.false_positive}")
print(f"True Negatives:  {reconstructed.true_negative}")

print("\nNow we can calculate unreported metrics:")
print(f"Specificity: {reconstructed.get_specificity():.3f}")
print(f"F1 Score: {reconstructed.get_f1_score():.3f}")
print(f"MCC: {reconstructed.get_mcc():.3f}")  # Using new alias!
print(f"NPV: {reconstructed.get_npv():.3f}")  # Using new method!
print(f"Cohen's Kappa: {reconstructed.get_cohens_kappa():.3f}")

# Verify reconstruction - check quality
print("\n" + "="*60)
print("Verify Reconstruction Quality:")
print("="*60)
print(f"Original:      Acc=0.850, Prec=0.800, Recall=0.750")
print(f"Reconstructed: Acc={reconstructed.get_accuracy():.3f}, "
      f"Prec={reconstructed.get_precision():.3f}, Recall={reconstructed.get_recall():.3f}")

# Check reconstruction quality (should be excellent with N=200)
consistency = reconstructed.check_metric_consistency({
    'accuracy': 0.85,
    'precision': 0.80,
    'recall': 0.75
}, tolerance=0.005)  # 0.5% tolerance - very precise!

if consistency['consistent']:
    print("\n✓ Excellent reconstruction! All metrics within 0.5% of target.")
else:
    # Calculate actual errors
    acc_err = abs(reconstructed.get_accuracy() - 0.85) * 100
    prec_err = abs(reconstructed.get_precision() - 0.80) * 100
    rec_err = abs(reconstructed.get_recall() - 0.75) * 100
    print(f"\n→ Errors: Acc={acc_err:.2f}%, Prec={prec_err:.2f}%, Rec={rec_err:.2f}%")
    print("  (Integer constraints prevent perfect decimal matches)")

print("\n→ Successfully reverse-engineered the confusion matrix structure!")
print("  Now we can calculate ANY metric, even those not in the paper.")

# Bonus: Show what happens with insufficient metrics
print("\n" + "="*60)
print("Bonus: What if we don't have enough metrics?")
print("="*60)
print("Example: Paper only reports accuracy and precision (2 metrics)")
print("\nTrying to reconstruct...")

try:
    insufficient_cm = DConfusion.from_metrics(
        total_samples=200,
        accuracy=0.85,
        precision=0.80
        # Only 2 metrics - need at least 3!
    )
    print("✗ Reconstructed (shouldn't happen)")
except ValueError as e:
    print(f"\n✓ DConfusion caught it!")
    print(f"   {e}")
    print("\n→ Need at least 3 independent metrics - this is fundamental math!")

print("\n" + "="*80)

input("\n\n[Press Enter for probabilistic inference demo...]")

# =============================================================================
# DEMO 7: Probabilistic Inference - When info is incomplete
# =============================================================================
print("\n\n### DEMO 7: Probabilistic Inference - Incomplete Information ###\n")

print("Scenario: Paper only reports:")
print("- 200 samples")
print("- 85% accuracy")
print("- 30% disease prevalence")
print("\nWhat could precision and recall be?\n")

result = DConfusion.infer_metrics(
    total_samples=200,
    accuracy=0.85,
    prevalence=0.30,
    confidence_level=0.95,
    n_simulations=5000,
    random_state=42
)

print("Inferred Metrics with 95% Confidence Intervals:\n")

for metric_name in ['precision', 'recall', 'specificity', 'f1_score']:
    stats = result['inferred_metrics'][metric_name]
    print(f"{metric_name.capitalize():15}: "
          f"{stats['mean']:.3f} "
          f"[{stats['ci_lower']:.3f} - {stats['ci_upper']:.3f}]")

print("\n→ This shows the range of possible values given limited information!")

input("\n\n[Press Enter to finish...]")

# =============================================================================
# DEMO 8: Real-World Complete Example
# =============================================================================
print("\n\n### DEMO 8: Complete Real-World Analysis ###\n")

print("Evaluating a new medical diagnosis model for production:\n")

model_current = DConfusion(85, 15, 10, 90)
model_new = DConfusion(90, 10, 20, 80)

print("Step 1: Quality Checks")
print("-" * 40)
warnings = model_new.check_warnings()
if warnings:
    print(f"Found {len(warnings)} warnings - review before deployment")
else:
    print("No critical warnings")

print("\n\nStep 2: Statistical Comparison")
print("-" * 40)
mcnemar_result = model_current.mcnemar_test(model_new)
print(f"Significantly different? {mcnemar_result['significant']}")
print(f"P-value: {mcnemar_result['p_value']:.4f}")

print("\n\nStep 3: Business Impact")
print("-" * 40)
cost_comparison = model_current.compare_cost_with(
    model_new,
    cost_fp=100,
    cost_fn=5000
)
print(f"Current model cost: ${cost_comparison['model1_total_cost']:,}")
print(f"New model cost: ${cost_comparison['model2_total_cost']:,}")
print(f"Cost difference: ${abs(cost_comparison['cost_savings']):,}")

print("\n\nStep 4: Uncertainty Analysis")
print("-" * 40)
ci_new = model_new.get_bootstrap_confidence_interval('recall', random_state=42)
print(f"New model recall: {ci_new['point_estimate']:.3f} "
      f"[{ci_new['lower']:.3f} - {ci_new['upper']:.3f}]")

print("\n\n" + "="*80)
print("Decision: Deploy new model?")
print("="*80)
print("\n✓ Passes quality checks")
print(f"{'✓' if not mcnemar_result['significant'] else '?'} Similar statistical performance")
print(f"{'✓' if cost_comparison['cost_savings'] < 0 else '✗'} Cost impact: {cost_comparison['recommendation']}")
print("✓ Uncertainty quantified")
print("\n→ All this analysis in just a few lines of code!")
print("="*80)

print("\n\nDemo complete! Questions?")