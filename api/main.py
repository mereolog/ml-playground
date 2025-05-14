import logging

from fastapi import FastAPI, WebSocket, HTTPException

from pydantic_schemas.configs.linear_regression_configs import LinearRegressionParams
from pydantic_schemas.interface.algorithm_interface import AlgorithmInfo

from typing import Dict, List, Type

app = FastAPI(
    title="ML Playground API",
    description="API to fetch configurations for machine learning algorithms.",
    version="1.0.0"
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
            description="A simple linear regression model."
        ),
        LinearRegressionParams
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
