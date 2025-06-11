from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AlgorithmInstanceResponse(BaseModel):
    session_id: str
    algorithm_name: str
    config: Dict[str, Any]
    status: str



class TrainingResponse(BaseModel):
    session_id: str
    status: str
    message: str
    training_history: Optional[Dict[str, Any]] = None