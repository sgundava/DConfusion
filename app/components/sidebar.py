"""Sidebar input components for DConfusion Streamlit App."""

import streamlit as st
from dconfusion import DConfusion, WarningSeverity
from utils.session import add_matrix, clear_all_matrices, get_matrices


def add_matrix_with_feedback(matrix_name, cm):
    """
    Add a confusion matrix to session state and show appropriate feedback.

    Args:
        matrix_name: Name identifier for the matrix
        cm: DConfusion instance
    """
    add_matrix(matrix_name, cm)

    # Check for warnings
    warnings = cm.check_warnings(include_info=False)
    critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
    warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

    if critical:
        st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(critical)} CRITICAL warning(s)")
    elif warning_level:
        st.sidebar.warning(f"⚠️ Added {matrix_name} with {len(warning_level)} warning(s)")
    else:
        st.sidebar.success(f"✅ Added {matrix_name}")


def render_sidebar():
    """Render the sidebar with matrix input options."""
    st.sidebar.header("Add Confusion Matrix")

    input_method = st.sidebar.radio(
        "Input Method",
        ["Binary (TP/FN/FP/TN)", "Multi-class Matrix", "From Predictions"]
    )

    # Use model counter for auto-increment
    default_name = f"Model {st.session_state.model_counter}"
    matrix_name = st.sidebar.text_input(
        "Matrix Name",
        value=default_name,
        key=f"matrix_name_input_{st.session_state.model_counter}"
    )

    if input_method == "Binary (TP/FN/FP/TN)":
        _render_binary_input(matrix_name)
    elif input_method == "Multi-class Matrix":
        _render_multiclass_input(matrix_name)
    else:  # From Predictions
        _render_predictions_input(matrix_name)

    # Clear all button
    if st.sidebar.button("🗑️ Clear All Matrices"):
        clear_all_matrices()
        st.rerun()


def _render_binary_input(matrix_name):
    """Render binary confusion matrix input."""
    st.sidebar.markdown("### Enter Values")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        tp = st.number_input("True Positive", min_value=0, value=85, step=1)
        fn = st.number_input("False Negative", min_value=0, value=15, step=1)

    with col2:
        fp = st.number_input("False Positive", min_value=0, value=10, step=1)
        tn = st.number_input("True Negative", min_value=0, value=90, step=1)

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            cm = DConfusion(tp, fn, fp, tn)
            add_matrix_with_feedback(matrix_name, cm)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")


def _render_multiclass_input(matrix_name):
    """Render multi-class confusion matrix input."""
    st.sidebar.markdown("### Multi-class Configuration")
    n_classes = st.sidebar.number_input("Number of Classes", min_value=2, max_value=10, value=3, step=1)

    st.sidebar.markdown("### Enter Matrix Values")
    matrix_values = []

    for i in range(n_classes):
        row = []
        cols = st.sidebar.columns(n_classes)
        for j in range(n_classes):
            with cols[j]:
                val = st.number_input(
                    f"[{i},{j}]",
                    min_value=0,
                    value=10 if i == j else 2,
                    step=1,
                    key=f"matrix_{i}_{j}"
                )
                row.append(val)
        matrix_values.append(row)

    labels_input = st.sidebar.text_input("Labels (comma-separated, optional)", value="")
    labels = [l.strip() for l in labels_input.split(",")] if labels_input else None

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            cm = DConfusion(confusion_matrix=matrix_values, labels=labels)
            add_matrix_with_feedback(matrix_name, cm)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")


def _render_predictions_input(matrix_name):
    """Render predictions-based input."""
    st.sidebar.markdown("### Enter Predictions")
    y_true_input = st.sidebar.text_area("True Labels (comma-separated)", value="1,0,1,1,0,0,1,0")
    y_pred_input = st.sidebar.text_area("Predicted Labels (comma-separated)", value="1,0,0,1,0,1,1,0")

    if st.sidebar.button("Add Matrix", type="primary"):
        try:
            y_true = [int(x.strip()) for x in y_true_input.split(",")]
            y_pred = [int(x.strip()) for x in y_pred_input.split(",")]
            cm = DConfusion.from_predictions(y_true, y_pred)
            add_matrix_with_feedback(matrix_name, cm)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
