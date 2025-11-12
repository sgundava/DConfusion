"""Session state management for DConfusion Streamlit App."""

import streamlit as st


def init_session_state():
    """Initialize session state for storing confusion matrices."""
    if 'matrices' not in st.session_state:
        st.session_state.matrices = {}


def add_matrix(matrix_name, cm):
    """
    Add a confusion matrix to session state.

    Args:
        matrix_name: Name identifier for the matrix
        cm: DConfusion instance
    """
    st.session_state.matrices[matrix_name] = cm


def remove_matrix(matrix_name):
    """
    Remove a confusion matrix from session state.

    Args:
        matrix_name: Name identifier for the matrix to remove
    """
    if matrix_name in st.session_state.matrices:
        del st.session_state.matrices[matrix_name]


def clear_all_matrices():
    """Clear all matrices from session state."""
    st.session_state.matrices = {}


def get_matrices():
    """
    Get all matrices from session state.

    Returns:
        dict: Dictionary of matrix_name -> DConfusion instances
    """
    return st.session_state.matrices
