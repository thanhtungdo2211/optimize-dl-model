import sys
from ultralytics import YOLO
import cv2
import torch

# Add quantization path
sys.path.insert(1, '.')
import quantization.quantize as quantize

if __name__ == "__main__":
    yolo_model  = YOLO("yolov8s.pt")  # Load a pre-trained YOLOv8 model

    model_path = "/yolov9/yolo_deepstream/yolov8/qat_yolov8s_v1.pt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    quantize.initialize()

    ckpt = torch.load(model_path, map_location=device)
    qat_model = ckpt['model'] if 'model' in ckpt else ckpt
    qat_model = qat_model.to(device).float().eval()

    quantize.replace_custom_module_forward_yolov8(qat_model)


    setattr(yolo_model, 'model', qat_model)  # Ensure model is set correctly

    yolo_model.export(format="onnx")



