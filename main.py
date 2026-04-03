from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Tree-AI backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # YOLO import csak itt, hogy ne foglaljon memóriát induláskor
    from ultralytics import YOLO

    # Modell betöltése (kicsi modell)
    model = YOLO("yolov8n.pt")

    # Kép beolvasása
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Predikció
    results = model(image)

    # Eredmények konvertálása
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })

    return JSONResponse({"detections": detections})
