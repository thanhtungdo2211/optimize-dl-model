import sys
import cv2
import torch
import numpy as np

# Add quantization path
sys.path.insert(1, '.')
import quantize as quantize

def simple_inference(model_path, image_path, output_path="result.jpg"):
    """Simple inference function with fixes"""
    
    # Initialize quantization
    quantize.initialize()
    
    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ckpt = torch.load(model_path, map_location=device)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model = model.to(device).float().eval()
    
    # Apply quantization
    quantize.replace_custom_module_forward_yolov8(model)

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return None
        
    original = img.copy()
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")

    # Preprocess với letterbox như YOLOv8
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    # Apply letterbox
    img_resized, ratio, (dw, dh) = letterbox(img, (640, 640))
    print(f"Letterbox info: ratio={ratio:.3f}, padding=({dw:.1f}, {dh:.1f})")
    
    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_rgb.transpose(2, 0, 1)  # HWC to CHW
    img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    print(f"Input tensor shape: {img_tensor.shape}")

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
    
    print(f"Model output type: {type(pred)}")
    if isinstance(pred, (list, tuple)):
        print(f"Number of outputs: {len(pred)}")
        for i, p in enumerate(pred):
            if isinstance(p, torch.Tensor):
                print(f"Output {i} shape: {p.shape}")
    
    # Process output
    if isinstance(pred, (list, tuple)):
        pred = pred[0]  # Take first output
    
    if len(pred.shape) == 3:
        pred = pred.transpose(1, 2)  # [1, 84, 8400] -> [1, 8400, 84]
        pred = pred[0]  # Remove batch dimension -> [8400, 84]
        
    print(f"Prediction shape after processing: {pred.shape}")
    
    # Get class scores
    class_scores = pred[:, 4:]  # [8400, 80] for COCO
    max_scores, class_preds = torch.max(class_scores, dim=1)
    
    print(f"Max score range: [{max_scores.min():.3f}, {max_scores.max():.3f}]")
    
    # Filter by confidence
    conf_threshold = 0.25
    conf_mask = max_scores > conf_threshold
    filtered_pred = pred[conf_mask]
    filtered_scores = max_scores[conf_mask]
    filtered_classes = class_preds[conf_mask]
    
    print(f"Number of detections after confidence filtering: {len(filtered_pred)}")
    
    if len(filtered_pred) == 0:
        print("No detections found!")
        cv2.imwrite(output_path, original)
        return original
    
    # QUAN TRỌNG: Model output đã ở pixel coordinates (640x640), không phải normalized!
    # Get boxes - đã ở format pixel trên 640x640
    boxes = filtered_pred[:, :4]  # x,y,w,h in pixel coordinates on 640x640
    
    print("DEBUG: First 3 raw detections (pixel coordinates on 640x640):")
    for i in range(min(3, len(filtered_pred))):
        box = filtered_pred[i, :4]
        score = filtered_scores[i]
        cls = filtered_classes[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], score={score:.3f}, class={cls}")
    
    # Apply NMS
    from torchvision.ops import nms
    
    # Convert to xyxy format (vẫn trên 640x640)
    boxes_xyxy_640 = boxes.clone()
    boxes_xyxy_640[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy_640[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy_640[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy_640[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    print("DEBUG: First 3 detections after xyxy conversion (640x640 scale):")
    for i in range(min(3, len(boxes_xyxy_640))):
        box = boxes_xyxy_640[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Apply NMS
    nms_indices = nms(boxes_xyxy_640, filtered_scores, iou_threshold=0.45)
    
    # Keep only NMS survivors
    final_boxes_640 = boxes_xyxy_640[nms_indices]
    final_scores = filtered_scores[nms_indices]
    final_classes = filtered_classes[nms_indices]
    
    print(f"Final detections after NMS: {len(final_boxes_640)}")
    
    if len(final_boxes_640) == 0:
        print("No detections after NMS!")
        cv2.imwrite(output_path, original)
        return original
    
    print("DEBUG: Final detections after NMS (640x640 scale):")
    for i in range(len(final_boxes_640)):
        box = final_boxes_640[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Scale back to original image size
    final_boxes = final_boxes_640.clone()
    
    # Remove padding first
    final_boxes[:, [0, 2]] -= dw  # remove width padding
    final_boxes[:, [1, 3]] -= dh  # remove height padding
    
    print(f"DEBUG: After removing padding (dw={dw:.1f}, dh={dh:.1f}):")
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Scale back to original size
    final_boxes /= ratio
    
    print(f"DEBUG: After scaling by ratio (1/{ratio:.3f}):")
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Clip to image bounds
    final_boxes[:, [0, 2]] = torch.clamp(final_boxes[:, [0, 2]], 0, w)
    final_boxes[:, [1, 3]] = torch.clamp(final_boxes[:, [1, 3]], 0, h)
    
    print(f"DEBUG: After clipping to image bounds (w={w}, h={h}):")
    for i in range(len(final_boxes)):
        box = final_boxes[i]
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # COCO class names
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Draw results
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    detection_count = 0
    for i, (box, cls_id, conf) in enumerate(zip(final_boxes, final_classes, final_scores)):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cls_id = int(cls_id.cpu().numpy())
        conf = float(conf.cpu().numpy())
        
        # Ensure valid box coordinates
        if x2 <= x1 or y2 <= y1:
            print(f"WARNING: Invalid box {i+1}: [{x1}, {y1}, {x2}, {y2}]")
            continue
            
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        class_name = coco_names[cls_id] if cls_id < len(coco_names) else f"class{cls_id}"
        label = f'{class_name}: {conf:.2f}'
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(original, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(original, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        print(f"Detection {i+1}: {class_name} ({conf:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
        detection_count += 1
    
    # Save result
    cv2.imwrite(output_path, original)
    print(f"Result saved to {output_path}")
    print(f"Successfully drew {detection_count} detections")
    return original

# Example usage
if __name__ == "__main__":
    model_path = "/yolov9/yolo_deepstream/yolov8/qat_yolov8s_v1.pt"
    image_path = "/yolov9/yolo_deepstream/yolov8/ultralytics/assets/zidane.jpg"
    
    result = simple_inference(model_path, image_path)
    
    if result is not None:
        print("✅ Inference completed!")
        print("📁 Check result.jpg for output")
    else:
        print("❌ Inference failed!")