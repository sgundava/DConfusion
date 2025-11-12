"""Helper functions for DConfusion Streamlit App."""

import tempfile
import os


def create_temp_csv(cm, name):
    """
    Create a temporary CSV file for download.

    Args:
        cm: DConfusion instance
        name: Name for the file

    Returns:
        str: CSV file content
    """
    tmpfile = tempfile.mktemp(suffix='.csv')
    cm.to_csv(tmpfile)
    with open(tmpfile, 'r') as f:
        csv_data = f.read()
    os.remove(tmpfile)
    return csv_data
