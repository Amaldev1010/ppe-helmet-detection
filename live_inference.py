import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque

# ── CONFIG ──────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\DELL\ppe_project\best.onnx"
CONF_THRESH = 0.30
NMS_THRESH  = 0.40
INPUT_SIZE  = 640
CLASSES = ["head", "helmet"]
COLORS  = [(0, 0, 220), (0, 200, 0)]  # green = helmet, red = no helmet
# ────────────────────────────────────────────────────────

print("Loading model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model loaded! Opening webcam...")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

fps_list = []
detection_buffer = deque(maxlen=3)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame!")
        break

    h, w = frame.shape[:2]

    # ── PRE-PROCESSING ──────────────────────────────────
    t0 = time.perf_counter()
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    pre_ms = (time.perf_counter() - t0) * 1000

    # ── INFERENCE ──────────────────────────────────────
    t1 = time.perf_counter()
    outputs = session.run(None, {input_name: img})
    infer_ms = (time.perf_counter() - t1) * 1000

    # ── POST-PROCESSING / NMS ───────────────────────────
    t2 = time.perf_counter()
    preds = outputs[0][0].T
    boxes_raw, scores_raw, class_ids = [], [], []

    for row in preds:
        confs = row[4:]
        cls_id = int(np.argmax(confs))
        conf = float(confs[cls_id])
        if conf < CONF_THRESH:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw / 2) * w / INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / INPUT_SIZE)
        x2 = int((cx + bw / 2) * w / INPUT_SIZE)
        y2 = int((cy + bh / 2) * h / INPUT_SIZE)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        boxes_raw.append([x1, y1, x2 - x1, y2 - y1])
        scores_raw.append(conf)
        class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(boxes_raw, scores_raw, CONF_THRESH, NMS_THRESH)
    post_ms = (time.perf_counter() - t2) * 1000

    # ── SMOOTHING BUFFER ────────────────────────────────
    current_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            current_detections.append({
                'box': boxes_raw[i],
                'score': scores_raw[i],
                'class_id': class_ids[i]
            })
    detection_buffer.append(current_detections)

    # Show detections only if seen in at least 2 of last 3 frames
    stable_detections = current_detections if len([d for d in detection_buffer if len(d) > 0]) >= 2 else []

    # ── DRAW BOXES ──────────────────────────────────────
    for det in stable_detections:
        x, y, bw, bh = det['box']
        color = COLORS[det['class_id'] % len(COLORS)]
        label = f"{CLASSES[det['class_id']]} {det['score']:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.rectangle(frame, (x, y - 24), (x + len(label) * 9, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── FPS (excluding render time) ─────────────────────
    total_no_render = pre_ms + infer_ms + post_ms
    fps = 1000.0 / max(total_no_render, 0.001)
    fps_list.append(fps)

    # ── OVERLAY METRICS ─────────────────────────────────
    cv2.rectangle(frame, (0, 0), (330, 115), (0, 0, 0), -1)
    metrics = [
        f"FPS (excl. render) : {fps:.1f}",
        f"Pre-process        : {pre_ms:.1f} ms",
        f"Inference          : {infer_ms:.1f} ms",
        f"Post-process (NMS) : {post_ms:.1f} ms",
    ]
    for idx, line in enumerate(metrics):
        cv2.putText(frame, line, (8, 25 + idx * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # ── SHOW FRAME ──────────────────────────────────────
    cv2.imshow("PPE Helmet Detection — Edge Inference (FP16 ONNX)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if fps_list:
    print(f"\n✅ Session complete")
    print(f"   Avg FPS : {np.mean(fps_list):.1f}")
    print(f"   Max FPS : {np.max(fps_list):.1f}")
    print(f"   Min FPS : {np.min(fps_list):.1f}")