"""
Visualizations Tab Component

Displays confusion matrix visualizations in a grid layout.
"""

import streamlit as st
from dconfusion import WarningSeverity


def render_visualizations_tab(matrices):
    """
    Render the visualizations tab showing confusion matrices in a grid.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
    st.header("Confusion Matrix Visualizations")

    # Options for visualization
    col1, col2 = st.columns(2)
    with col1:
        normalize = st.checkbox("Normalize (show percentages)", value=False)
    with col2:
        show_metrics = st.checkbox("Show metrics panel (binary only)", value=True)

    # Display matrices in columns
    n_matrices = len(matrices)
    cols_per_row = min(3, n_matrices)

    matrix_items = list(matrices.items())
    for i in range(0, n_matrices, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (name, cm) in enumerate(matrix_items[i:i+cols_per_row]):
            with cols[j]:
                # Check for warnings and add status badge
                warnings = cm.check_warnings(include_info=False)
                critical = [w for w in warnings if w.severity == WarningSeverity.CRITICAL]
                warning_level = [w for w in warnings if w.severity == WarningSeverity.WARNING]

                if critical:
                    st.subheader(f"🔴 {name}")
                    st.caption(f"{len(critical)} critical issue(s)")
                elif warning_level:
                    st.subheader(f"🟡 {name}")
                    st.caption(f"{len(warning_level)} warning(s)")
                else:
                    st.subheader(f"🟢 {name}")

                try:
                    fig = cm.plot(
                        normalize=normalize,
                        show_metrics=show_metrics and cm.n_classes == 2,
                        figsize=(6, 5)
                    )
                    st.pyplot(fig)

                    # Remove button
                    if st.button(f"Remove", key=f"remove_{name}"):
                        del st.session_state.matrices[name]
                        st.rerun()
                except Exception as e:
                    st.error(f"Error plotting: {str(e)}")
