from fastapi import WebSocket
from typing import Dict
import logging

class ConnectionManager:
    """Manages WebSocket connections for real-time training updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logging.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove session mapping if exists
        for session_id, conn_id in list(self.session_connections.items()):
            if conn_id == connection_id:
                del self.session_connections[session_id]
        logging.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_session(self, session_id: str, data: dict):
        """Send data to WebSocket connection associated with a session."""
        connection_id = self.session_connections.get(session_id)
        if connection_id and connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_json(data)
            except Exception as e:
                logging.error(f"Failed to send to session {session_id}: {e}")
                self.disconnect(connection_id)
    
    def associate_session(self, session_id: str, connection_id: str):
        """Associate a session with a WebSocket connection."""
        self.session_connections[session_id] = connection_id
        logging.info(f"Associated session {session_id} with connection {connection_id}")

