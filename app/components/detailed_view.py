"""
Detailed View Tab Component

Displays detailed information, properties, and export options for each confusion matrix.
"""

import streamlit as st
import tempfile
import os


def render_detailed_view_tab(matrices):
    """
    Render the detailed matrix view tab.

    Args:
        matrices: Dictionary of {name: DConfusion} confusion matrices
    """
    st.header("Detailed Matrix View")

    for name, cm in matrices.items():
        with st.expander(f"📋 {name}", expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Confusion Matrix**")
                st.text(str(cm))

                st.markdown("**Frequency Distribution**")
                st.text(cm.frequency())

            with col2:
                st.markdown("**Matrix Properties**")
                st.write(f"- Classes: {cm.n_classes}")
                st.write(f"- Labels: {cm.labels}")
                st.write(f"- Total Samples: {cm.total}")
                st.write(f"- Correct Predictions: {cm.get_sum_of_corrects()}")
                st.write(f"- Incorrect Predictions: {cm.get_sum_of_errors()}")

                if cm.n_classes == 2:
                    st.markdown("**Binary Values**")
                    binary_vals = cm.get_binary_values()
                    for key, val in binary_vals.items():
                        st.write(f"- {key}: {val}")

            # Export options
            st.markdown("**Export Options**")
            col1, col2 = st.columns(2)

            with col1:
                dict_export = cm.to_dict()
                st.download_button(
                    label="📥 Export as JSON",
                    data=str(dict_export),
                    file_name=f"{name.replace(' ', '_')}.json",
                    mime="application/json",
                    key=f"json_{name}"
                )

            with col2:
                tmpfile = tempfile.mktemp(suffix='.csv')
                cm.to_csv(tmpfile)
                with open(tmpfile, 'r') as f:
                    csv_data = f.read()
                os.remove(tmpfile)

                st.download_button(
                    label="📥 Export as CSV",
                    data=csv_data,
                    file_name=f"{name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"csv_{name}"
                )
