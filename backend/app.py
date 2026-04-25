from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import os

# ========================
# INIT APP
# ========================
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# ========================
# LOAD YOLO MODEL
# ========================
print("🚀 Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("✅ Model loaded")

TARGET_CLASSES = ["cat", "dog", "person"]

# ========================
# PREDICT ROUTE (FIXED)
# ========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🔥 PREDICT CALLED")

        # ✅ Safe file access
        file = request.files.get("file")
        if file is None:
            print("❌ No file received")
            return jsonify([])

        img_bytes = file.read()

        # ✅ Convert bytes → image
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Image decode failed")
            return jsonify([])

        # ✅ Run YOLO
        results = model(img, conf=0.25)

        print("📦 Boxes:", results[0].boxes)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

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

        print("✅ Detections:", detections)

        return jsonify(detections)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify([])

# ========================
# SERVE FRONTEND
# ========================
@app.route("/")
def serve_frontend():
    return app.send_static_file("index.html")

# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
