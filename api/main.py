import logging

from fastapi import FastAPI, WebSocket

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

connected_clients = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_json()
        logging.info("Received data: %s", data)
