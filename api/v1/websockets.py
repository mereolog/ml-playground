import logging
from typing import List

from algorithms_registry import ALGORITHM_REGISTRY
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from shared import algorithm_sessions, manager
from pydantic import ValidationError
from schemas.interface.requests import (
    AlgorithmInstanceRequest,
    StreamingTrainingRequest,
    TrainingRequest,
)
from schemas.interface.responses import AlgorithmInstanceResponse, TrainingResponse

router = APIRouter()


@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """Handle WebSocket connections for real-time training updates."""
    await manager.connect(websocket, connection_id)
    try:
        while True:
            data = await websocket.receive_json()
            logging.info("Received WebSocket data: %s", data)

            # Echo back for now, but you can add more sophisticated handling
            await websocket.send_json({"type": "echo", "data": data})

    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        manager.disconnect(connection_id)

