import logging
from typing import List

from fastapi import APIRouter, HTTPException
from main import algorithm_sessions, app, manager
from schemas.interface.requests import StreamingTrainingRequest, TrainingRequest
from schemas.interface.responses import TrainingResponse

router = APIRouter()


@router.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about an algorithm session."""
    if session_id not in algorithm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = algorithm_sessions[session_id]
    return {
        "session_id": session_id,
        "algorithm_name": session["algorithm_name"],
        "config": session["config"],
        "status": session["status"],
        "trained": session["trained"],
    }


@router.post("/sessions/{session_id}/train", response_model=TrainingResponse)
async def train_algorithm(session_id: str, request: TrainingRequest):
    """Train an algorithm instance with provided data (non-streaming)."""
    if session_id not in algorithm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = algorithm_sessions[session_id]
    algorithm_instance = session["algorithm_instance"]

    try:
        import numpy as np

        # Convert to numpy arrays
        X_array = np.array(request.X)
        y_array = np.array(request.y)

        logging.info(
            f"Training algorithm {session_id} with data shape: {X_array.shape}"
        )

        # Train the algorithm
        algorithm_instance.fit(X_array, y_array)

        # Update session status
        session["status"] = "trained"
        session["trained"] = True

        # Get training history if available
        training_history = None
        if hasattr(algorithm_instance, "get_training_history"):
            training_history = algorithm_instance.get_training_history()

        logging.info(f"Successfully trained algorithm {session_id}")

        return TrainingResponse(
            session_id=session_id,
            status="trained",
            message="Algorithm trained successfully",
            training_history=training_history,
        )

    except Exception as e:
        session["status"] = "error"
        logging.error(f"Training failed for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/sessions/{session_id}/train_streaming")
async def train_algorithm_streaming(request: StreamingTrainingRequest):
    """Train an algorithm instance with real-time streaming updates."""
    if request.session_id not in algorithm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = algorithm_sessions[request.session_id]
    algorithm_instance = session["algorithm_instance"]

    # Associate session with WebSocket connection
    manager.associate_session(request.session_id, request.connection_id)

    # Set up streaming callback for the algorithm
    async def streaming_callback(data: dict):
        await manager.send_to_session(request.session_id, data)

    try:
        import numpy as np

        # Convert to numpy arrays
        X_array = np.array(request.X)
        y_array = np.array(request.y)

        logging.info(f"Starting streaming training for session {request.session_id}")

        # Enable streaming on the algorithm instance
        if hasattr(algorithm_instance, "enable_streaming"):
            algorithm_instance.enable_streaming(streaming_callback)

            # Train with streaming (if supported)
            if hasattr(algorithm_instance, "fit_streaming"):
                await algorithm_instance.fit_streaming(X_array, y_array)
            else:
                # Fallback to regular training
                algorithm_instance.fit(X_array, y_array)
        else:
            # Algorithm doesn't support streaming, use regular training
            algorithm_instance.fit(X_array, y_array)

        # Update session status
        session["status"] = "trained"
        session["trained"] = True

        logging.info(
            f"Successfully completed streaming training for session {request.session_id}"
        )

        return {
            "message": "Streaming training completed",
            "session_id": request.session_id,
        }

    except Exception as e:
        session["status"] = "error"
        logging.error(
            f"Streaming training failed for session {request.session_id}: {str(e)}"
        )

        # Send error via WebSocket
        await manager.send_to_session(
            request.session_id, {"task": "training_error", "error": str(e)}
        )

        raise HTTPException(
            status_code=500, detail=f"Streaming training failed: {str(e)}"
        )


@router.post("/sessions/{session_id}/predict")
async def predict(session_id: str, X: List[List[float]]):
    """Make predictions using a trained algorithm."""
    if session_id not in algorithm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = algorithm_sessions[session_id]

    if not session["trained"]:
        raise HTTPException(status_code=400, detail="Algorithm not trained yet")

    try:
        import numpy as np

        algorithm_instance = session["algorithm_instance"]
        X_array = np.array(X)

        predictions = algorithm_instance.predict(X_array)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete an algorithm session."""
    if session_id not in algorithm_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del algorithm_sessions[session_id]
    logging.info(f"Deleted session: {session_id}")

    return {"message": "Session deleted successfully"}

