"""FastAPI application providing endpoints for machine learning algorithm configurations.

This module serves as the main entry point for the ML Playground API, providing endpoints
to list available algorithms, fetch their configurations, and handle WebSocket connections
for real-time interactions.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from algorithms.supervised.linear_regression import LinearRegression
from connection_manager import ConnectionManager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from v1 import algorithms, sessions, websockets

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ML Playground API",
    description="API to fetch configurations for machine learning algorithms.",
    version="1.0.0",
)


app.include_router(algorithms.router, prefix="/algorithms", tags=["algorithms"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(websockets.router, prefix="/ws", tags=["websockets"])


# Global connection manager
manager = ConnectionManager()

# Global registry for algorithm sessions
algorithm_sessions: Dict[str, Dict[str, Any]] = {}

