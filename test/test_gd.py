import os
import sys

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from core.base import LogisticRegressionGD

# Test the LogisticRegressionGD class
def test_logistic_regression_gd():
    # Create an instance of the LogisticRegressionGD class
    log_reg = LogisticRegressionGD()

    # Intialize X and y
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 1, 0, 1, 0]).reshape(-1, 1)

    # Fit the model
    log_reg.fit(X, y)
