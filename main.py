import base64
import io
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

face_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    logger.info("Loading InsightFace model...")
    from insightface.app import FaceAnalysis
    face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("Model ready.")
    yield

app = FastAPI(title="Boletix Face Compare Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://boletix.vercel.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    image1_base64: str
    image2_base64: str


def b64_to_bgr(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img.thumbnail((640, 640))
    arr = np.array(img)
    return arr[:, :, ::-1].copy()  # RGB to BGR for InsightFace


def enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Histogram equalization to help detection on dark/low-contrast images."""
    import cv2
    yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def detect_cedula_faces(img_bgr: np.ndarray):
    """Try harder to detect face on ID card — small face within larger image."""
    faces = face_app.get(img_bgr)
    if faces:
        return faces
    # Retry with contrast enhancement for dark/low-quality images
    faces = face_app.get(enhance_contrast(img_bgr))
    if faces:
        return faces
    # Retry on upper half of card (face is usually there on Colombian IDs)
    h = img_bgr.shape[0]
    faces = face_app.get(img_bgr[:h // 2, :])
    return faces


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": face_app is not None}


@app.post("/compare")
def compare(req: CompareRequest):
    if face_app is None:
        raise HTTPException(503, {"error": "model_loading", "message": "Model not ready yet"})

    try:
        img1 = b64_to_bgr(req.image1_base64)
        img2 = b64_to_bgr(req.image2_base64)
    except Exception as e:
        raise HTTPException(400, {"error": "invalid_image", "message": str(e)})

    faces1 = detect_cedula_faces(img1)
    faces2 = face_app.get(img2)

    if not faces1:
        raise HTTPException(422, {
            "error": "no_face_cedula",
            "message": "No se detectó un rostro en la foto de la Cédula."
        })
    if not faces2:
        raise HTTPException(422, {
            "error": "no_face_selfie",
            "message": "No se detectó un rostro en el selfie."
        })

    emb1 = faces1[0].normed_embedding
    emb2 = faces2[0].normed_embedding

    cosine_sim = float(np.dot(emb1, emb2))
    similarity = round(max(0.0, min(100.0, cosine_sim * 100)), 1)
    verified = cosine_sim > 0.3

    logger.info(f"compare: cosine={cosine_sim:.4f} similarity={similarity} verified={verified}")

    return {
        "verified": verified,
        "similarity": similarity,
        "distance": round(1 - cosine_sim, 4),
    }
