import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import onnx

# YOLOv7 imports
sys.path.append('./')
from models.experimental import attempt_load, End2End
from models.yolo import Model
from models.common import Conv
from utils.general import check_img_size, set_logging, colorstr
from utils.torch_utils import select_device
from utils.google_utils import attempt_download

warnings.filterwarnings("ignore")


def load_pruned_model(weight, device):
    """Load pruned YOLOv7 model"""
    attempt_download(weight)
    
    # Load model from checkpoint
    if weight.endswith('.pt'):
        ckpt = torch.load(weight, map_location=device)
        if 'model' in ckpt:
            model = ckpt['model']  # From training checkpoint
        else:
            model = ckpt  # Direct model save
    else:
        raise ValueError(f"Unsupported weight format: {weight}")
    
    # Ensure model compatibility
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    model.float()
    model.eval()
    
    # Fuse model for inference
    with torch.no_grad():
        model.fuse()
    
    return model


def export_pruned_end2end_onnx(model, file, img_size=640, dynamic_batch=True, topk_all=100, 
                               simplify=False, iou_thres=0.45, conf_thres=0.25):
    """Export pruned model to end2end ONNX format"""
    
    device = next(model.parameters()).device
    model.float()
    
    # Grid size and image size verification
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    image_check = [img_size, img_size]
    img_size, _ = [check_img_size(x, gs) for x in image_check]  # verify img_size are gs-multiples
    
    # Create dummy input
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    
    print(f"Input shape: {dummy.shape}")
    print(f"Model stride: {model.stride}")
    
    # Configure model for end2end export
    model.model[-1].export = False  # set Detect() layer grid export
    
    # Handle grid function for ONNX compatibility
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())
    
    labels = model.names
    batch_size = 'batch' if dynamic_batch else 1
    
    # Dynamic axes configuration
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
        }
        output_axes = {
            'num_dets': {0: 'batch'},
            'det_boxes': {0: 'batch'},
            'det_scores': {0: 'batch'},
            'det_classes': {0: 'batch'},
        }
        dynamic_axes.update(output_axes)
    else:
        dynamic_axes = None
    
    print('\nStarting export end2end onnx model for TensorRT...')
    
    # Wrap model with End2End module
    end2end_model = End2End(model, topk_all, iou_thres, conf_thres, None, device, len(labels))
    
    # Output configuration
    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    shapes = [batch_size, 1, batch_size, topk_all, 4, batch_size, topk_all, batch_size, topk_all]
    
    # Export to ONNX
    torch.onnx.export(
        end2end_model, 
        dummy, 
        file, 
        verbose=False,
        opset_version=13,
        input_names=["images"], 
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Load and modify ONNX model
    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model)
    
    # Set output shapes for dynamic batch
    if dynamic_batch:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))
    
    # Simplify ONNX if requested
    if simplify:
        try:
            import onnxsim
            print('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'ONNX simplification failed'
            print('ONNX simplification successful')
        except Exception as e:
            print(f'ONNX simplification failed: {e}')
    
    # Save final ONNX model
    onnx.save(onnx_model, file)
    print(f'ONNX export success, saved as {file}')
    
    # Restore original grid function
    model.model[-1]._make_grid = grid_old_func
    
    return file


def export_standard_onnx(model, file, img_size=640, dynamic_batch=True):
    """Export pruned model to standard ONNX format (without end2end)"""
    
    device = next(model.parameters()).device
    model.float()
    
    gs = max(int(model.stride.max()), 32)
    image_check = [img_size, img_size]
    img_size, _ = [check_img_size(x, gs) for x in image_check]
    
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    
    # Configure for standard export
    model.model[-1].concat = True
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())
    
    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"}, 
            "outputs": {0: "batch"}
        }
    
    print('\nStarting standard ONNX export...')
    
    torch.onnx.export(
        model, 
        dummy, 
        file, 
        verbose=False,
        opset_version=13,
        input_names=["images"], 
        output_names=["outputs"],
        dynamic_axes=dynamic_axes
    )
    
    # Restore settings
    model.model[-1].concat = False
    model.model[-1]._make_grid = grid_old_func
    
    print(f'Standard ONNX export success, saved as {file}')
    return file


def main():
    parser = argparse.ArgumentParser(description='Export pruned YOLOv7 model to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to pruned model weights (.pt file)')
    parser.add_argument('--output', type=str, help='Output ONNX file path')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for export')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--end2end', action='store_true', help='Export end2end ONNX model')
    parser.add_argument('--dynamic-batch', action='store_true', help='Export with dynamic batch size')
    parser.add_argument('--topk-all', type=int, default=100, help='Max detections per image for end2end')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for NMS')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    
    args = parser.parse_args()
    
    # Setup
    set_logging()
    device = select_device(args.device)
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = Path(args.weights).stem
        suffix = '_end2end' if args.end2end else '_standard'
        args.output = f"{base_name}_pruned{suffix}.onnx"
    
    print(f"{colorstr('Export:')} Starting export with torch {torch.__version__}...")
    print(f"Loading pruned model from: {args.weights}")
    
    # Load pruned model
    model = load_pruned_model(args.weights, device)
    
    print(f"Model loaded successfully")
    print(f"Model classes: {len(model.names)} {model.names}")
    print(f"Model stride: {model.stride}")
    
    # Export based on mode
    if args.end2end:
        export_pruned_end2end_onnx(
            model=model,
            file=args.output,
            img_size=args.img_size,
            dynamic_batch=args.dynamic_batch,
            topk_all=args.topk_all,
            simplify=args.simplify,
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres
        )
        print(f"\n{colorstr('End2End Export:')} Model exported with NMS included")
        print(f"Output format: [num_dets, det_boxes, det_scores, det_classes]")
    else:
        export_standard_onnx(
            model=model,
            file=args.output,
            img_size=args.img_size,
            dynamic_batch=args.dynamic_batch
        )
        print(f"\n{colorstr('Standard Export:')} Model exported without NMS")
        print(f"Output format: [outputs] - requires post-processing")
    
    print(f"\n{colorstr('Export complete:')} {args.output}")
    print(f"Model size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()