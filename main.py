from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import subprocess
import sys

app = FastAPI()

def ensure_ultralytics():
    try:
        import ultralytics
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.1.0"])
    finally:
        from ultralytics import YOLO
        return YOLO

@app.get("/")
def root():
    return {"status": "Tree-AI backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    YOLO = ensure_ultralytics()
    model = YOLO("yolov8n.pt")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    return JSONResponse({"detections": detections})
