import base64
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="AI-Based Emotion Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = None
if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    loaded = cv2.CascadeClassifier(cascade_path)
    if loaded is not None and not loaded.empty():
        face_cascade = loaded


class PredictRequest(BaseModel):
    image: str


def decode_base64_image(data_url: str) -> np.ndarray:
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    img_bytes = base64.b64decode(payload)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image frame")
    return frame


def estimate_emotions(gray_face: np.ndarray) -> Dict[str, float]:
    resized = cv2.resize(gray_face, (96, 96))
    h, w = resized.shape
    lower = resized[int(h * 0.55) : h, :]
    upper = resized[0 : int(h * 0.45), :]
    center = resized[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]

    contrast = float(np.std(center))
    brightness = float(np.mean(resized))
    lower_bright = float(np.mean(lower))
    upper_bright = float(np.mean(upper))

    happy_score = max(0.05, min(1.0, (lower_bright - upper_bright + 25.0) / 50.0))
    sad_score = max(0.05, min(1.0, (135.0 - brightness) / 90.0))
    angry_score = max(0.05, min(1.0, (upper_bright - lower_bright + 20.0) / 50.0))
    surprised_score = max(0.05, min(1.0, (contrast - 25.0) / 45.0))
    neutral_score = max(0.20, min(1.0, 1.0 - abs(contrast - 40.0) / 60.0))
    fear_score = max(0.05, min(0.8, (surprised_score + sad_score) / 2.2))
    disgust_score = max(0.05, min(0.8, (angry_score + sad_score) / 2.4))

    raw_scores = {
        "Happy": happy_score,
        "Sad": sad_score,
        "Angry": angry_score,
        "Surprised": surprised_score,
        "Disgust": disgust_score,
        "Fear": fear_score,
        "Neutral": neutral_score,
    }
    total = sum(raw_scores.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw_scores.items()}


@app.get("/")
def serve_home() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "index.html")


@app.post("/api/predict")
def predict_emotion(req: PredictRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        frame = decode_base64_image(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes: List[List[int]] = []

    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
        )
        for (x, y, w, h) in faces:
            boxes.append([int(x), int(y), int(w), int(h)])
    else:
        h, w = gray.shape
        bw, bh = int(w * 0.45), int(h * 0.55)
        bx, by = int((w - bw) / 2), int((h - bh) / 2)
        boxes.append([bx, by, bw, bh])

    detected_faces: List[Dict[str, Any]] = []
    for (x, y, w, h) in boxes:
        roi_gray = gray[y : y + h, x : x + w]
        if roi_gray.size == 0:
            roi_gray = gray
        probs = estimate_emotions(roi_gray)
        top_emotion = max(probs, key=probs.get)
        detected_faces.append(
            {
                "box": [x, y, w, h],
                "emotion": top_emotion,
                "confidence": probs[top_emotion],
                "probabilities": probs,
            }
        )

    return {
        "faces": detected_faces,
        "latency_ms": round((time.perf_counter() - start) * 1000.0, 1),
    }
