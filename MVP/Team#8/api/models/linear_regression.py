from typing import Any, Dict

import numpy as np
from configs.model_parameters_configs import LinearRegressionParams
from models.base_model import SupervisedAlgorithm


class LinearRegression(SupervisedAlgorithm):
    """Linear Regression implementation."""

    def __init__(self, params: LinearRegressionParams = None):
        super().__init__(params or LinearRegressionParams())
        self.model = None
        self.history = {"loss": []}
