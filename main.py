# ============================
# CORS engedélyezés
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fejlesztéshez jó, később szűkíthető
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# YOLO modell betöltése
# ============================
MODEL_URL = "https://huggingface.co/regenhuhu/tree-ai-model/resolve/main/yolov8n.pt"
model = YOLO(MODEL_URL)

@app.get("/")
def root():
    return {"status": "Tree-AI backend running"}

# ============================
# /predict végpont (frontend ezt hívja)
# ============================
@app.post("/predict")
async def predict(conf: float = 0.1, file: UploadFile = File(...)):
    # ideiglenes fájl létrehozása
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    # YOLO futtatása
    results = model(temp_path, conf=conf)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "class": cls,
            "confidence": conf
        })

    os.remove(temp_path)

    # dummy ajánlások és terv (később bővíthető)
    return {
        "detections": detections,
        "recommendations": [],
        "plan": []
    }

# ============================
# Railway indítás
# ============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
from fastapi import FastAPI, UploadFile, File
import httpx

app = FastAPI()

YOLO_API_URL = "https://tree-ai-yolo-api.onrender.com/predict"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # A képet továbbküldjük a YOLO API-nak
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(YOLO_API_URL, files=files)

    # Ha a YOLO API hibát dob
    if response.status_code != 200:
        return {"error": "YOLO API error", "details": response.text}

    # YOLO válasz vissza a frontendnek
    return response.json()
