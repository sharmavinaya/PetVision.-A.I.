from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ========================
# Load YOLOv8 Model
# ========================
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

TARGET_CLASSES = ["cat", "dog", "person"]

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name not in TARGET_CLASSES:
                continue

            if confidence < 0.25:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

    return jsonify(detections)

# ✅ Serve frontend
@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
