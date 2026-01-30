"""
Consistency Testing Tab Component

Provides consistency testing for binary classification metrics using mlscorecheck.
This wraps the work of Fazekas & Kovács (2024) to verify if reported scores
could mathematically result from a given experimental setup.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import dconfusion
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from dconfusion import (
        is_consistency_testing_available,
        get_mlscorecheck_version,
        check_consistency,
        check_consistency_kfold,
    )
    _CONSISTENCY_AVAILABLE = is_consistency_testing_available()
except ImportError:
    _CONSISTENCY_AVAILABLE = False


def render_consistency_testing_tab(matrices):
    """
    Render the consistency testing tab.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
    st.header("🔬 Consistency Testing")

    st.markdown("""
    Verify if reported performance scores are **mathematically consistent** with a given
    experimental setup. This helps detect impossible results or reporting errors in
    published research.

    *Powered by [mlscorecheck](https://github.com/FalseNegativeLab/mlscorecheck)
    by Fazekas & Kovács (2024)*
    """)

    if not _CONSISTENCY_AVAILABLE:
        st.warning("""
        ⚠️ **Consistency testing requires mlscorecheck**

        Install with:
        ```bash
        pip install mlscorecheck
        ```
        Or:
        ```bash
        pip install dconfusion[consistency]
        ```
        """)
        return

    # Show version info
    version = get_mlscorecheck_version()
    if version:
        st.caption(f"mlscorecheck version: {version}")

    # Two modes: from loaded matrices or manual input
    mode = st.radio(
        "Testing Mode",
        ["Check Loaded Model", "Manual Input"],
        horizontal=True,
        help="Use loaded confusion matrices or enter values manually"
    )

    if mode == "Check Loaded Model":
        _render_model_based_testing(matrices)
    else:
        _render_manual_testing()


def _render_model_based_testing(matrices):
    """Render consistency testing using loaded confusion matrices."""

    if not matrices:
        st.info("👈 Add a confusion matrix using the sidebar first, or use Manual Input mode.")
        return

    # Filter to binary models only
    binary_models = {name: cm for name, cm in matrices.items() if cm.is_binary}

    if not binary_models:
        st.warning("Consistency testing requires binary classification models. Add a binary confusion matrix to use this feature.")
        return

    model_names = list(binary_models.keys())
    selected_model = st.selectbox("Select Model", model_names, key="consistency_model")
    cm = binary_models[selected_model]

    st.markdown("---")

    # Two sub-options: verify own scores or check external scores
    test_type = st.radio(
        "What would you like to test?",
        ["Verify Model's Own Scores (Sanity Check)", "Check External/Published Scores"],
        help="Either verify this model's scores pass consistency, or check if some external scores are possible"
    )

    if test_type == "Verify Model's Own Scores (Sanity Check)":
        _render_self_verification(cm, selected_model)
    else:
        _render_external_score_check(cm, selected_model)


def _render_self_verification(cm, model_name):
    """Verify the model's own scores pass consistency testing."""

    st.markdown(f"""
    This will verify that **{model_name}**'s own metrics pass consistency testing.
    This should always pass for a valid confusion matrix - if it fails, there may be a bug.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Setup:** p={cm.p} positives, n={cm.n} negatives")
    with col2:
        st.markdown(f"**Total samples:** {cm.p + cm.n}")

    if st.button("Run Self-Verification", key="verify_own"):
        with st.spinner("Verifying..."):
            try:
                result = cm.verify_own_scores(decimal_places=4)

                if result.is_consistent:
                    st.success("✅ **Self-verification passed!** The model's scores are mathematically consistent.")
                else:
                    st.error("❌ **Self-verification failed!** This indicates an unexpected issue.")

                with st.expander("Details", expanded=True):
                    st.json({
                        "is_consistent": result.is_consistent,
                        "scores_tested": result.scores_tested,
                        "epsilon": result.epsilon,
                        "message": result.message
                    })

            except Exception as e:
                st.error(f"Error during verification: {str(e)}")


def _render_external_score_check(cm, model_name):
    """Check if external/published scores are consistent with the model's setup."""

    st.markdown(f"""
    Enter reported scores to check if they're mathematically possible given **{model_name}**'s
    experimental setup (p={cm.p} positives, n={cm.n} negatives).

    **Use case:** Verify if metrics reported in a paper could actually result from this sample size.
    """)

    st.markdown("#### Enter Reported Scores")
    st.caption("Enter at least one score. Use the same metric names as reported.")

    col1, col2, col3 = st.columns(3)

    with col1:
        acc = st.number_input("Accuracy (acc)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="check_acc", help="Leave empty to skip")
        sens = st.number_input("Sensitivity/Recall (sens)", min_value=0.0, max_value=1.0, value=None,
                              format="%.4f", key="check_sens")

    with col2:
        spec = st.number_input("Specificity (spec)", min_value=0.0, max_value=1.0, value=None,
                              format="%.4f", key="check_spec")
        ppv = st.number_input("Precision/PPV (ppv)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="check_ppv")

    with col3:
        npv = st.number_input("NPV (npv)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="check_npv")
        f1 = st.number_input("F1 Score (f1)", min_value=0.0, max_value=1.0, value=None,
                            format="%.4f", key="check_f1")

    epsilon = st.number_input(
        "Epsilon (numerical tolerance)",
        min_value=0.0001, max_value=0.1, value=0.0001,
        format="%.4f",
        help="Tolerance for rounding. If scores are reported to 2 decimal places, use 0.005"
    )

    # Build scores dict
    scores = {}
    if acc is not None:
        scores['acc'] = acc
    if sens is not None:
        scores['sens'] = sens
    if spec is not None:
        scores['spec'] = spec
    if ppv is not None:
        scores['ppv'] = ppv
    if npv is not None:
        scores['npv'] = npv
    if f1 is not None:
        scores['f1'] = f1

    if not scores:
        st.info("Enter at least one score to check.")
        return

    if st.button("Check Consistency", key="check_external"):
        with st.spinner("Checking consistency..."):
            try:
                result = cm.check_reported_scores(scores, epsilon=epsilon)

                if result.is_consistent:
                    st.success(f"""
                    ✅ **Scores are CONSISTENT**

                    The reported scores could mathematically result from an experiment with
                    {cm.p} positives and {cm.n} negatives.
                    """)
                else:
                    st.error(f"""
                    ❌ **Scores are INCONSISTENT**

                    The reported scores are **mathematically impossible** given the experimental setup
                    (p={cm.p}, n={cm.n}). This may indicate:
                    - Reporting errors
                    - Rounding issues (try increasing epsilon)
                    - Different experimental setup than claimed
                    """)

                with st.expander("Details", expanded=True):
                    st.json({
                        "is_consistent": result.is_consistent,
                        "p": result.p,
                        "n": result.n,
                        "scores_tested": result.scores_tested,
                        "epsilon": result.epsilon,
                        "message": result.message
                    })

            except Exception as e:
                st.error(f"Error checking consistency: {str(e)}")


def _render_manual_testing():
    """Render manual consistency testing without loaded matrices."""

    st.markdown("""
    Enter the experimental setup and reported scores to check if they're mathematically consistent.

    **Use case:** Verify if metrics reported in a paper could result from the claimed sample sizes.
    """)

    st.markdown("#### Experimental Setup")

    col1, col2 = st.columns(2)
    with col1:
        p = st.number_input("Number of Positives (p)", min_value=1, value=50,
                           help="Actual positive samples in the dataset")
    with col2:
        n = st.number_input("Number of Negatives (n)", min_value=1, value=50,
                           help="Actual negative samples in the dataset")

    st.caption(f"Total samples: {p + n}")

    st.markdown("---")
    st.markdown("#### Reported Scores")
    st.caption("Enter at least one score. Common abbreviations: acc, sens, spec, ppv, npv, f1")

    col1, col2, col3 = st.columns(3)

    with col1:
        acc = st.number_input("Accuracy (acc)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="manual_acc")
        sens = st.number_input("Sensitivity/Recall (sens)", min_value=0.0, max_value=1.0, value=None,
                              format="%.4f", key="manual_sens")

    with col2:
        spec = st.number_input("Specificity (spec)", min_value=0.0, max_value=1.0, value=None,
                              format="%.4f", key="manual_spec")
        ppv = st.number_input("Precision/PPV (ppv)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="manual_ppv")

    with col3:
        npv = st.number_input("NPV (npv)", min_value=0.0, max_value=1.0, value=None,
                             format="%.4f", key="manual_npv")
        f1 = st.number_input("F1 Score (f1)", min_value=0.0, max_value=1.0, value=None,
                            format="%.4f", key="manual_f1")

    epsilon = st.number_input(
        "Epsilon (numerical tolerance)",
        min_value=0.0001, max_value=0.1, value=0.0001,
        format="%.4f", key="manual_epsilon",
        help="Tolerance for rounding. If scores are reported to 2 decimal places, use 0.005"
    )

    # Build scores dict
    scores = {}
    if acc is not None:
        scores['acc'] = acc
    if sens is not None:
        scores['sens'] = sens
    if spec is not None:
        scores['spec'] = spec
    if ppv is not None:
        scores['ppv'] = ppv
    if npv is not None:
        scores['npv'] = npv
    if f1 is not None:
        scores['f1'] = f1

    if not scores:
        st.info("Enter at least one score to check.")
        return

    if st.button("Check Consistency", key="manual_check"):
        with st.spinner("Checking consistency..."):
            try:
                result = check_consistency(p=p, n=n, scores=scores, epsilon=epsilon)

                if result.is_consistent:
                    st.success(f"""
                    ✅ **Scores are CONSISTENT**

                    The reported scores could mathematically result from an experiment with
                    {p} positives and {n} negatives.
                    """)
                else:
                    st.error(f"""
                    ❌ **Scores are INCONSISTENT**

                    The reported scores are **mathematically impossible** given the experimental setup
                    (p={p}, n={n}). This may indicate:
                    - Reporting errors
                    - Rounding issues (try increasing epsilon)
                    - Different experimental setup than claimed
                    """)

                with st.expander("Details", expanded=True):
                    st.json({
                        "is_consistent": result.is_consistent,
                        "p": result.p,
                        "n": result.n,
                        "scores_tested": result.scores_tested,
                        "epsilon": result.epsilon,
                        "message": result.message
                    })

            except Exception as e:
                st.error(f"Error checking consistency: {str(e)}")

    # K-fold cross-validation section
    st.markdown("---")
    st.subheader("📊 K-Fold Cross-Validation Consistency")

    with st.expander("Check K-Fold CV Scores", expanded=False):
        st.markdown("""
        Check if reported k-fold cross-validation scores are consistent with the dataset.
        Supports both Mean of Scores (MoS) and Score of Means (SoM) aggregation.
        """)

        col1, col2 = st.columns(2)
        with col1:
            kfold_p = st.number_input("Total Positives", min_value=1, value=50, key="kfold_p")
            kfold_n = st.number_input("Total Negatives", min_value=1, value=50, key="kfold_n")
        with col2:
            k_folds = st.number_input("Number of Folds (k)", min_value=2, max_value=20, value=5, key="k_folds")
            aggregation = st.selectbox("Aggregation Method", ["mos", "som"],
                                       help="MoS: Mean of Scores per fold. SoM: Score of aggregated counts.",
                                       key="kfold_agg")

        st.markdown("##### Reported Scores")
        col1, col2 = st.columns(2)
        with col1:
            kfold_acc = st.number_input("Accuracy", min_value=0.0, max_value=1.0, value=None,
                                       format="%.4f", key="kfold_acc")
            kfold_sens = st.number_input("Sensitivity", min_value=0.0, max_value=1.0, value=None,
                                        format="%.4f", key="kfold_sens")
        with col2:
            kfold_spec = st.number_input("Specificity", min_value=0.0, max_value=1.0, value=None,
                                        format="%.4f", key="kfold_spec")

        kfold_epsilon = st.number_input("Epsilon", min_value=0.0001, max_value=0.1, value=0.0001,
                                       format="%.4f", key="kfold_epsilon")

        kfold_scores = {}
        if kfold_acc is not None:
            kfold_scores['acc'] = kfold_acc
        if kfold_sens is not None:
            kfold_scores['sens'] = kfold_sens
        if kfold_spec is not None:
            kfold_scores['spec'] = kfold_spec

        if kfold_scores and st.button("Check K-Fold Consistency", key="check_kfold"):
            with st.spinner("Checking k-fold consistency (this may take a moment)..."):
                try:
                    result = check_consistency_kfold(
                        p=kfold_p, n=kfold_n, k=k_folds,
                        scores=kfold_scores,
                        aggregation=aggregation,
                        epsilon=kfold_epsilon
                    )

                    if result.is_consistent:
                        st.success(f"✅ K-fold scores are **CONSISTENT** with the dataset.")
                    else:
                        st.error(f"❌ K-fold scores are **INCONSISTENT** - mathematically impossible.")

                    with st.expander("Details"):
                        st.write(f"Aggregation: {aggregation.upper()}")
                        st.write(f"Folds: {k_folds}")
                        st.json({
                            "is_consistent": result.is_consistent,
                            "message": result.message
                        })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # About section
    with st.expander("ℹ️ About Consistency Testing", expanded=False):
        st.markdown("""
        ### What is Consistency Testing?

        Consistency testing verifies if reported performance scores are **mathematically possible**
        given the experimental setup (number of positive and negative samples).

        **Key Insight:** Given p positives and n negatives, only certain combinations of
        TP, FP, TN, FN are valid. This constrains which metric values are achievable.

        ### When to Use

        - **Verifying Published Results:** Check if metrics in a paper make sense
        - **Detecting Reporting Errors:** Find typos or calculation mistakes
        - **Quality Assurance:** Validate your own pipelines
        - **Peer Review:** Identify impossible claims

        ### How It Works

        1. Takes the experimental setup (p, n) and reported scores
        2. Uses interval arithmetic to handle rounding uncertainty (epsilon)
        3. Checks if any valid confusion matrix could produce those scores
        4. Returns boolean: consistent or impossible

        ### Attribution

        This feature wraps **mlscorecheck** by Fazekas & Kovács (2024):

        > *"Testing the Numerical Consistency of Reported Machine Learning Performance Scores"*

        - [GitHub: mlscorecheck](https://github.com/FalseNegativeLab/mlscorecheck)
        - Handles single test sets, k-fold CV, and complex experimental designs
        - Uses interval arithmetic and integer linear programming

        ### Comparison with DConfusion's Metric Inference

        | Feature | Consistency Testing | Metric Inference |
        |---------|---------------------|------------------|
        | **Question** | "Are these scores possible?" | "What values fit these constraints?" |
        | **Output** | Boolean (yes/no) | Values + confidence intervals |
        | **Use Case** | Validation, verification | Reconstruction, estimation |
        | **Approach** | Exact mathematical bounds | Monte Carlo simulation |

        Both are complementary - use consistency testing to validate, and metric inference to explore.
        """)
