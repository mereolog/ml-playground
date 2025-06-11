import logging
import uuid
from typing import List

from algorithms_registry import ALGORITHM_REGISTRY
from fastapi import APIRouter, HTTPException
from main import algorithm_sessions, app, manager
from pydantic import ValidationError
from schemas.interface.algorithm_interface import AlgorithmInfo
from schemas.interface.requests import AlgorithmInstanceRequest
from schemas.interface.responses import AlgorithmInstanceResponse

router = APIRouter()

@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_available_algorithms():
    """List all available machine learning algorithms."""
    return [entry.info for entry in ALGORITHM_REGISTRY.values()]


@router.get("/algorithms/{algorithm_name}/config_schema")
async def get_algorithm_config_schema(algorithm_name: str):
    """Retrieve the configuration schema for a specific algorithm."""
    entry = ALGORITHM_REGISTRY.get(algorithm_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    return entry.pydantic_model.model_json_schema()


@router.post("/algorithms/create", response_model=AlgorithmInstanceResponse)
async def create_algorithm_instance(request: AlgorithmInstanceRequest):
    """Create a new algorithm instance with given configuration."""
    # Validate algorithm exists
    entry = ALGORITHM_REGISTRY.get(request.algorithm_name)
    if not entry:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # Check if algorithm is implemented
    if entry.algorithm_class is None:
        raise HTTPException(
            status_code=501, 
            detail=f"Algorithm '{request.algorithm_name}' not implemented yet"
        )
    
    try:
        # Validate configuration using Pydantic model
        config_instance = entry.pydantic_model(**request.config)
        
        # Create algorithm instance
        session_id = str(uuid.uuid4())
        algorithm_instance = entry.algorithm_class(params=config_instance)
        
        # Store in session
        algorithm_sessions[session_id] = {
            "algorithm_name": request.algorithm_name,
            "algorithm_instance": algorithm_instance,
            "config": config_instance.model_dump(),
            "status": "created",
            "trained": False
        }
        
        logging.info(f"Created algorithm instance: {session_id} ({request.algorithm_name})")
        
        return AlgorithmInstanceResponse(
            session_id=session_id,
            algorithm_name=request.algorithm_name,
            config=config_instance.model_dump(),
            status="created"
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create algorithm: {str(e)}")


