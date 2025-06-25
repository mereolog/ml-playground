"""FastAPI application providing endpoints for machine learning algorithm configurations.

This module serves as the main entry point for the ML Playground API, providing endpoints
to list available algorithms, fetch their configurations, and handle WebSocket connections
for real-time interactions.
"""

import logging
from typing import Dict, List, Type

from fastapi import FastAPI, HTTPException, WebSocket

from schemas.configs.decisions_tree_config import DecisionTreeParams
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams
from schemas.configs.kmeans_configs import KMeansParams
from schemas.configs.linear_regression import LinearRegressionParams
from schemas.configs.logistic_regression import LogisticRegressionParams
from schemas.configs.polynomial_regression_configs import PolynomialRegressionParams
from schemas.interface.algorithm_interface import AlgorithmInfo

app = FastAPI(
    title="ML Playground API",
    description="API to fetch configurations for machine learning algorithms.",
    version="1.0.0",
)


class AlgorithmRegistryEntry:
    """Registry entry for ML algorithms containing metadata and configuration schema.

    Attributes:
        info: Algorithm metadata including name and description
        pydantic_model: Configuration schema class for the algorithm
    """

    def __init__(self, info: AlgorithmInfo, pydantic_model: Type):
        self.info = info
        self.pydantic_model = pydantic_model


ALGORITHM_REGISTRY: Dict[str, AlgorithmRegistryEntry] = {
    "linear_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="linear_regression",
            display_name="Linear Regression",
            description="A simple linear regression model.",
        ),
        LinearRegressionParams,
    ),
    "decision_tree": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="decision_tree",
            display_name="Decision Tree",
            description="A decision tree algorithm for classification and regression.",
        ),
        DecisionTreeParams,
    ),
    "k_nearest_neighbours": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="k_nearest_neighbours",
            display_name="K-Nearest Neighbours",
            description=(
                "A k-nearest neighbours algorithm for classification and regression."
            ),
        ),
        KNeighborsParams,
    ),
    "kmeans": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="kmeans",
            display_name="K-Means Clustering",
            description="A k-means clustering algorithm to partition data into k clusters.",
        ),
        KMeansParams,
    ),
    "logistic_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="logistic_regression",
            display_name="Logistic Regression",
            description="A logistic regression algorithm for binary classification.",
        ),
        LogisticRegressionParams,
    ),
    "polynomial_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="polynomial_regression",
            display_name="Polynomial Regression",
            description="A polynomial regression algorithm for non-linear relationships.",
        ),
        PolynomialRegressionParams,
    ),
}

# Configure logging
logging.basicConfig(level=logging.INFO)

connected_clients = []


@app.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_available_algorithms():
    """List all available machine learning algorithms.

    Returns:
        List[AlgorithmInfo]: List of algorithm metadata including names and descriptions
    """
    return [entry.info for entry in ALGORITHM_REGISTRY.values()]


@app.get("/algorithms/{algorithm_name}/config_schema")
async def get_algorithm_config_schema(algorithm_name: str):
    """Retrieve the configuration schema for a specific algorithm.

    Args:
        algorithm_name (str): Name of the algorithm to get configuration for

    Returns:
        dict: JSON schema for the algorithm's configuration

    Raises:
        HTTPException: If the specified algorithm is not found (404)
    """
    entry = ALGORITHM_REGISTRY.get(algorithm_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return entry.pydantic_model.model_json_schema()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time communication.

    Args:
        websocket (WebSocket): WebSocket connection instance
    """
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        logging.info("Received data: %s", data)