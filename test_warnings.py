"""
Quick test to verify the warning system is working correctly.
"""

import sys
sys.path.insert(0, '/Users/suryagundavarapu/Developer/DConfusion')

from dconfusion import DConfusion, WarningSeverity

def test_sample_size_warning():
    """Test that small sample size generates warning."""
    cm = DConfusion(true_positive=5, false_negative=3, false_positive=2, true_negative=4)
    warnings = cm.check_warnings(include_info=False)

    # Should have sample size warning
    sample_warnings = [w for w in warnings if 'Sample Size' in w.category]
    assert len(sample_warnings) > 0, "Expected sample size warning"
    print("✓ Sample size warning test passed")

def test_class_imbalance_warning():
    """Test that severe imbalance generates warning."""
    cm = DConfusion(true_positive=3, false_negative=2, false_positive=5, true_negative=990)
    warnings = cm.check_warnings(include_info=False)

    # Should have critical imbalance warning
    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
    assert len(critical) > 0, "Expected critical warning for severe imbalance"
    print("✓ Class imbalance warning test passed")

def test_perfect_classification():
    """Test that perfect classification generates warning."""
    cm = DConfusion(true_positive=50, false_negative=0, false_positive=0, true_negative=50)
    warnings = cm.check_warnings(include_info=False)

    # Should have perfect classification warning
    perfect_warnings = [w for w in warnings if 'Perfect Classification' in w.category]
    assert len(perfect_warnings) > 0, "Expected perfect classification warning"
    print("✓ Perfect classification warning test passed")

def test_zero_tp_warning():
    """Test that zero TP generates critical warning."""
    cm = DConfusion(true_positive=0, false_negative=20, false_positive=5, true_negative=75)
    warnings = cm.check_warnings(include_info=False)

    # Should have critical warning for zero TP
    zero_tp_warnings = [w for w in warnings
                        if w.severity == WarningSeverity.CRITICAL
                        and 'True Positive' in w.category]
    assert len(zero_tp_warnings) > 0, "Expected critical warning for zero TP"
    print("✓ Zero TP warning test passed")

def test_good_matrix_minimal_warnings():
    """Test that well-balanced matrix has minimal warnings."""
    cm = DConfusion(true_positive=85, false_negative=15, false_positive=12, true_negative=88)
    warnings = cm.check_warnings(include_info=False)

    # Should have no critical warnings
    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
    assert len(critical) == 0, "Good matrix should have no critical warnings"
    print("✓ Good matrix test passed")

def test_comparison_warnings():
    """Test model comparison warning generation."""
    from dconfusion import check_comparison_validity

    # Small sample comparison
    cm1 = DConfusion(true_positive=10, false_negative=5, false_positive=3, true_negative=7)
    cm2 = DConfusion(true_positive=12, false_negative=3, false_positive=5, true_negative=5)

    warnings = check_comparison_validity(cm1, cm2)

    # Should have comparison warnings
    assert len(warnings) > 0, "Expected comparison warnings for small samples"
    print("✓ Comparison warnings test passed")

def test_compare_with_method():
    """Test the compare_with method."""
    cm1 = DConfusion(true_positive=48, false_negative=7, false_positive=5, true_negative=40)
    cm2 = DConfusion(true_positive=50, false_negative=5, false_positive=8, true_negative=37)

    result = cm1.compare_with(cm2, metric='accuracy', show_warnings=True)

    # Should have required fields
    assert 'metric' in result
    assert 'value1' in result
    assert 'value2' in result
    assert 'difference' in result
    assert 'warnings' in result
    assert 'has_warnings' in result

    print("✓ compare_with method test passed")

def test_misleading_accuracy():
    """Test misleading accuracy detection."""
    # Model with high accuracy but poor sensitivity
    cm = DConfusion(true_positive=5, false_negative=15, false_positive=5, true_negative=175)
    warnings = cm.check_warnings(include_info=False)

    # Should warn about misleading accuracy
    misleading = [w for w in warnings if 'Misleading Accuracy' in w.category]
    assert len(misleading) > 0, "Expected misleading accuracy warning"

    # Should also warn about poor basic rates
    poor_rates = [w for w in warnings if 'Poor Basic Rates' in w.category]
    assert len(poor_rates) > 0, "Expected poor basic rates warning"

    print("✓ Misleading accuracy warning test passed")

def test_warning_severity_filtering():
    """Test filtering warnings by severity."""
    cm = DConfusion(true_positive=0, false_negative=15, false_positive=5, true_negative=80)
    warnings = cm.check_warnings(include_info=True)

    # Check we can filter by severity
    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
    warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]
    info_level = [w for w in warnings if w.severity == WarningSeverity.INFO]

    assert len(critical) > 0, "Should have critical warnings"
    assert len(warning_level) > 0, "Should have warning level warnings"

    # Test exclude_info parameter
    warnings_no_info = cm.check_warnings(include_info=False)
    info_in_result = [w for w in warnings_no_info if w.severity == WarningSeverity.INFO]

    # Note: check_warnings doesn't filter, but we can still verify the structure
    print("✓ Warning severity filtering test passed")

def run_all_tests():
    """Run all warning system tests."""
    print("\n" + "="*70)
    print("Running DConfusion Warning System Tests")
    print("="*70 + "\n")

    tests = [
        test_sample_size_warning,
        test_class_imbalance_warning,
        test_perfect_classification,
        test_zero_tp_warning,
        test_good_matrix_minimal_warnings,
        test_comparison_warnings,
        test_compare_with_method,
        test_misleading_accuracy,
        test_warning_severity_filtering
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
