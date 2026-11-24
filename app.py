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

EMOTIONS = ["Happy", "Sad", "Angry", "Surprised", "Disgust", "Fear", "Neutral"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class PredictRequest(BaseModel):
    image: str


def decode_base64_image(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("Empty image payload")
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    img_bytes = base64.b64decode(payload)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image frame")
    return frame


def estimate_emotions(gray_face: np.ndarray) -> Dict[str, float]:
    resized = cv2.resize(gray_face, (96, 96))
    h, w = resized.shape
    upper = resized[0 : int(h * 0.45), :]
    lower = resized[int(h * 0.55) : h, :]
    center = resized[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]

    contrast = float(np.std(center))
    brightness = float(np.mean(resized))
    lower_bright = float(np.mean(lower))
    upper_bright = float(np.mean(upper))

    smile_score = max(0.05, min(1.0, (lower_bright - upper_bright + 25.0) / 50.0))
    surprise_score = max(0.05, min(1.0, (contrast - 25.0) / 45.0))
    anger_score = max(0.05, min(1.0, (upper_bright - lower_bright + 20.0) / 50.0))
    sad_score = max(0.05, min(1.0, (135.0 - brightness) / 90.0))
    neutral_score = max(0.20, min(1.0, 1.0 - abs(contrast - 40.0) / 60.0))
    fear_score = max(0.05, min(0.8, (surprise_score + sad_score) / 2.2))
    disgust_score = max(0.05, min(0.8, (anger_score + sad_score) / 2.4))

    raw_scores = {
        "Happy": smile_score,
        "Sad": sad_score,
        "Angry": anger_score,
        "Surprised": surprise_score,
        "Disgust": disgust_score,
        "Fear": fear_score,
        "Neutral": neutral_score,
    }
    total = sum(raw_scores.values()) or 1.0
    return {k: round(v / total, 4) for k, v in raw_scores.items()}


@app.get("/")
def serve_home() -> FileResponse:
    index_path = Path(__file__).resolve().parent / "index.html"
    return FileResponse(index_path)


@app.post("/api/predict")
def predict_emotion(req: PredictRequest) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        frame = decode_base64_image(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
    )

    detected_faces: List[Dict[str, Any]] = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        probs = estimate_emotions(roi_gray)
        top_emotion = max(probs, key=probs.get)
        detected_faces.append(
            {
                "box": [int(x), int(y), int(w), int(h)],
                "emotion": top_emotion,
                "confidence": probs[top_emotion],
                "probabilities": probs,
            }
        )

    latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
    return {"faces": detected_faces, "latency_ms": latency_ms}
