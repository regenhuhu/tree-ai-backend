from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import os

app = FastAPI()

# CORS engedélyezés
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOLO modell betöltése HuggingFace-ről vagy URL-ről
MODEL_URL = "https://huggingface.co/regenhuhu/tree-ai-model/resolve/main/yolov8n.pt"
model = YOLO(MODEL_URL)

@app.get("/")
def root():
    return {"status": "Tree-AI backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "temp.jpg"

    with open(temp_path, "wb") as f:
        f.write(contents)

    results = model(temp_path)
    boxes = results[0].boxes.xyxy.tolist()

    os.remove(temp_path)
    return {"detections": boxes}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
