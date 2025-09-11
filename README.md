# DConfusion
A Python package for working with confusion matrices.

## Overview
`dconfusion` is a lightweight Python package that provides basic statistical functionality for working with confusion matrices. Confusion matrices are a crucial tool in machine learning and data analysis, allowing you to evaluate the performance of classification models.

## Features
* Basic confusion matrix representation
* Frequency calculation for each cell in the matrix

## Installation
You can install `dconfusion` using pip:

```bash
pip install dconfusion
```

Usage
Here's an example of how to use dconfusion:

```python
from dconfusion import *

cm = DConfusion(80, 70, 10, 20)

print(cm)

print(cm.frequency())

print(f"""Specificity: {cm.get_specificity():^15.2f}""")
print(f"""Sensitivity: {cm.get_sensitivity():^15.2f}""")
print(f"""Accuracy: {cm.get_accuracy():^15.2f}""")
print(f"""Precision: {cm.get_precision():^15.2f}""")
print(f"""F1 Score: {cm.get_f1_score():^15.2f}""")
print(f"""Matthews Correlation: {cm.get_matthews_correlation_coefficient():^15.2f}""")
```

# Roadmap
This is the initial release (v0.1) of dconfusion, and we plan to add more features in future releases. Some potential features include:
- Additional statistical metrics (e.g., accuracy, precision, recall, F1 score)
- Support for multi-class classification
- Integration with popular machine learning libraries

# Contributing
We welcome contributions to dconfusion! If you'd like to contribute, please fork the repository and submit a pull request.

# License
dconfusion is released under the MIT License. See LICENSE for details.

# Changelog
v0.1: Initial release with basic confusion matrix representation and frequency calculation