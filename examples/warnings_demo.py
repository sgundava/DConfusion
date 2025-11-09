"""
Demonstration of the DConfusion warning system.

This script shows how to use the warning system to identify potential
pitfalls in confusion matrix analysis based on research on binary
classification metrics.
"""

from dconfusion import DConfusion

print("="*80)
print("DConfusion Warning System Demonstration")
print("="*80)

# Example 1: Small sample size
print("\n\n### Example 1: Small Sample Size ###")
print("-" * 80)
cm1 = DConfusion(true_positive=8, false_negative=4, false_positive=3, true_negative=5)
print(cm1)
print(f"\nAccuracy: {cm1.get_accuracy():.3f}")
print("\nWarnings:")
cm1.print_warnings(include_info=False)

# Example 2: Severe class imbalance
print("\n\n### Example 2: Severe Class Imbalance ###")
print("-" * 80)
cm2 = DConfusion(true_positive=5, false_negative=3, false_positive=10, true_negative=982)
print(cm2)
print(f"\nAccuracy: {cm2.get_accuracy():.3f}")
print(f"Precision: {cm2.get_precision():.3f}")
print(f"Recall: {cm2.get_recall():.3f}")
print("\nWarnings:")
cm2.print_warnings(include_info=False)

# Example 3: Perfect classification (suspicious)
print("\n\n### Example 3: Perfect Classification ###")
print("-" * 80)
cm3 = DConfusion(true_positive=50, false_negative=0, false_positive=0, true_negative=50)
print(cm3)
print(f"\nAccuracy: {cm3.get_accuracy():.3f}")
print("\nWarnings:")
cm3.print_warnings(include_info=False)

# Example 4: Zero true positives
print("\n\n### Example 4: Zero True Positives ###")
print("-" * 80)
cm4 = DConfusion(true_positive=0, false_negative=20, false_positive=5, true_negative=75)
print(cm4)
print(f"\nAccuracy: {cm4.get_accuracy():.3f}")
print("\nWarnings:")
cm4.print_warnings(include_info=False)

# Example 5: Misleading high accuracy due to imbalance
print("\n\n### Example 5: Misleading High Accuracy ###")
print("-" * 80)
cm5 = DConfusion(true_positive=5, false_negative=15, false_positive=5, true_negative=175)
print(cm5)
print(f"\nAccuracy: {cm5.get_accuracy():.3f}")
print(f"Sensitivity: {cm5.get_recall():.3f}")
print(f"Specificity: {cm5.get_specificity():.3f}")
print("\nWarnings:")
cm5.print_warnings(include_info=False)

# Example 6: Good confusion matrix with reasonable sample size
print("\n\n### Example 6: Well-Balanced Confusion Matrix ###")
print("-" * 80)
cm6 = DConfusion(true_positive=85, false_negative=15, false_positive=12, true_negative=88)
print(cm6)
print(f"\nAccuracy: {cm6.get_accuracy():.3f}")
print(f"Precision: {cm6.get_precision():.3f}")
print(f"Recall: {cm6.get_recall():.3f}")
print(f"Specificity: {cm6.get_specificity():.3f}")
print(f"MCC: {cm6.get_matthews_correlation_coefficient():.3f}")
print("\nWarnings:")
cm6.print_warnings(include_info=False)

# Example 7: Comparing two models
print("\n\n### Example 7: Comparing Two Models ###")
print("-" * 80)
model_a = DConfusion(true_positive=48, false_negative=7, false_positive=5, true_negative=40)
model_b = DConfusion(true_positive=50, false_negative=5, false_positive=8, true_negative=37)

print("Model A:")
print(model_a)
print(f"Accuracy: {model_a.get_accuracy():.3f}")

print("\nModel B:")
print(model_b)
print(f"Accuracy: {model_b.get_accuracy():.3f}")

print("\nComparison:")
comparison = model_a.compare_with(model_b, metric='accuracy')
print(f"Difference in accuracy: {comparison['difference']:.4f}")
print(f"Better model: {comparison['better_model']}")

if comparison['has_warnings']:
    print("\nComparison Warnings:")
    for warning in comparison['warnings']:
        print(f"  {warning}")

# Example 8: Accessing warnings programmatically
print("\n\n### Example 8: Programmatic Access to Warnings ###")
print("-" * 80)
cm8 = DConfusion(true_positive=5, false_negative=5, false_positive=2, true_negative=8)
warnings = cm8.check_warnings(include_info=False)

print(f"Total warnings: {len(warnings)}")
for i, warning in enumerate(warnings, 1):
    print(f"\n{i}. [{warning.severity.value}] {warning.category}")
    print(f"   {warning.message}")
    if warning.recommendation:
        print(f"   → {warning.recommendation}")

# Check for critical warnings
critical = [w for w in warnings if w.severity.name == 'CRITICAL']
if critical:
    print(f"\n⚠ Found {len(critical)} critical warning(s)!")

print("\n" + "="*80)
print("End of demonstration")
print("="*80)
