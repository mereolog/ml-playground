from typing import Dict, Any
from connection_manager import ConnectionManager

manager = ConnectionManager()

# Global registry for algorithm sessions
algorithm_sessions: Dict[str, Dict[str, Any]] = {}