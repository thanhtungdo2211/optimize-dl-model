# run_inference.py
import time
import torch
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
# onnxruntime-gpu

# Config
MODEL_PATH = "yolov8s.onnx"
IMG_PATH = "ultralytics/assets/zidane.jpg"  # Change to your test image
INPUT_SIZE = (640, 640)
BATCH_SIZE = 1
WARMUP_RUNS = 5
TEST_RUNS = 50

def load_image(path, size):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def main():
    # Load ONNX model
    providers = [("TensorrtExecutionProvider", {
    "trt_max_workspace_size": 1 << 30,
    "trt_fp16_enable": True
}), "CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Load and prepare image
    img = load_image(IMG_PATH, INPUT_SIZE)

    # Warm-up (to avoid first-run overhead)
    for _ in range(WARMUP_RUNS):
        session.run(output_names, {input_name: img})

    # Timed runs
    times = []
    for _ in range(TEST_RUNS):
        start = time.time()
        session.run(output_names, {input_name: img})
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Average Inference Time: {np.mean(times):.2f} ms")
    print(f"FPS: {1000 / np.mean(times):.2f}")

if __name__ == "__main__":
    main()
