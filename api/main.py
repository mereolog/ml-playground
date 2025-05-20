import logging
from typing import Dict, List, Type

from fastapi import FastAPI, HTTPException, WebSocket
from schemas.configs.decision_tree_config import DecisionTreeParams
from schemas.configs.k_nearest_neighbour_config import KNeighborsParams
from schemas.configs.kmeans_config import KMeansParams
from schemas.configs.linear_regression_configs import LinearRegressionParams
from schemas.configs.logistic_regression_config import LogisticRegressionParams
from schemas.configs.polynomial_regression_configs import PolynomialRegressionParams
from schemas.interface.algorithm_interface import AlgorithmInfo

app = FastAPI(
    title="ML Playground API",
    description="API to fetch configurations for machine learning algorithms.",
    version="1.0.0",
)


class AlgorithmRegistryEntry:
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
            description="A k-nearest neighbours algorithm for classification and regression.",
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
    return [entry.info for entry in ALGORITHM_REGISTRY.values()]


@app.get("/algorithms/{algorithm_name}/config_schema")
async def get_algorithm_config_schema(algorithm_name: str):
    entry = ALGORITHM_REGISTRY.get(algorithm_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return entry.pydantic_model.model_json_schema()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        logging.info("Received data: %s", data)
