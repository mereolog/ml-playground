from typing import List, Dict, Any

from pydantic import BaseModel


class AlgorithmInstanceRequest(BaseModel):
    algorithm_name: str
    config: Dict[str, Any]


class StreamingTrainingRequest(BaseModel):
    session_id: str
    connection_id: str  # WebSocket connection ID
    X: List[List[float]]
    y: List[float]


class TrainingRequest(BaseModel):
    session_id: str
    X: List[List[float]]
    y: List[float]
