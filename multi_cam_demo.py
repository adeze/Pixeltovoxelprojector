import asyncio
import base64

import cv2
from fastapi import FastAPI, WebSocket

app = FastAPI()

CAMERA_URLS = [
    0,  # Local webcam
    "rtsp://user:pass@ip_address:554/stream",  # Example RTSP stream
    # Add more camera sources here
]


def get_frame(camera_index):
    cap = cv2.VideoCapture(CAMERA_URLS[camera_index])
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")
    return None


@app.websocket("/ws/camera/{camera_index}")
async def camera_stream(websocket: WebSocket, camera_index: int):
    await websocket.accept()
    while True:
        frame = get_frame(camera_index)
        if frame:
            await websocket.send_text(frame)
        await asyncio.sleep(0.03)  # ~30 FPS        await asyncio.sleep(0.03)  # ~30 FPS
