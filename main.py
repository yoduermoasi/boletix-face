import base64
import io
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Boletix Face Compare Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://boletix.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    image1_base64: str  # Cédula image
    image2_base64: str  # Selfie image


def b64_to_rgb(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    # Limit size to speed up processing
    img.thumbnail((800, 800))
    return np.array(img)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
def compare(req: CompareRequest):
    try:
        img1 = b64_to_rgb(req.image1_base64)
        img2 = b64_to_rgb(req.image2_base64)
    except Exception as e:
        raise HTTPException(400, {"error": "invalid_image", "message": str(e)})

    # Detect + encode faces (HOG model is faster on CPU than CNN)
    enc1 = face_recognition.face_encodings(img1, model="hog")
    enc2 = face_recognition.face_encodings(img2, model="hog")

    if not enc1:
        raise HTTPException(422, {
            "error": "no_face_cedula",
            "message": "No se detectó un rostro en la foto de la Cédula."
        })
    if not enc2:
        raise HTTPException(422, {
            "error": "no_face_selfie",
            "message": "No se detectó un rostro en el selfie."
        })

    # face_recognition distance: 0.0 = identical, 0.6 = different person threshold
    distance = float(face_recognition.face_distance([enc1[0]], enc2[0])[0])

    # Convert distance to 0-100 similarity score
    # distance 0.0 → 100%, distance 0.6 → 0%
    similarity = round(max(0.0, min(100.0, (1.0 - distance / 0.6) * 100)), 1)

    # Stricter threshold than default 0.6 — reduces false positives
    verified = distance < 0.45

    logger.info(f"compare: distance={distance:.4f} similarity={similarity} verified={verified}")

    return {
        "verified": verified,
        "similarity": similarity,
        "distance": round(distance, 4),
    }
