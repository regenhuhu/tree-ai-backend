from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# A modell lazy módon töltődik be, csak az első kérésnél
model = None

def get_model():
    global model
    if model is None:
        model = YOLO("yolov8n.pt")  # automatikusan letölti, ha nincs meg
    return model


@app.get("/")
def root():
    return {"status": "Tree-AI backend running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Kép beolvasása
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Modell betöltése
    model = get_model()

    # Predikció
    results = model(image)

    # YOLO eredmények konvertálása JSON-ra
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    return JSONResponse({"detections": detections})
