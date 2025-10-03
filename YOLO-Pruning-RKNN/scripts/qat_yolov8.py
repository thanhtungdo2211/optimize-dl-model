################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import sys
import os
import yaml
import warnings
import argparse
import json
from pathlib import Path
from copy import deepcopy

# QAT modules
import sys
sys.path.insert(1, '.')
import quantization.quantize as quantize

# PyTorch
import torch
import torch.nn as nn

# Ultralytics YOLOv8 imports
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
# from ultralytics.utils.checks import check_imgsz
# from ultralytics.utils.files import increment_path
# from ultralytics.data.utils import check_dataset
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel

from copy import deepcopy

# Disable warnings
warnings.filterwarnings("ignore")

ARGS = None  # Global variable to hold YOLOv8 args

class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)

def load_yolov8_model(weight, device):
    """Load YOLOv8 model from Ultralytics"""
    try:
        # # Load using Ultralytics YOLO
        # yolo_model = YOLO(weight)
        # model = yolo_model.model
        
        # print(model.names)
        
        # # Move to device and set properties
        # model = model.to(device)
        # model.float()
        # model.eval()
        
        # # # Fuse layers for optimization
        # model.fuse()
        
        # LOGGER.info(f"✅ Loaded YOLOv8 model: {weight}")
        
        from ultralytics.nn.tasks import attempt_load_weights

        model = attempt_load_weights(
            weight, device=device, inplace=True, fuse=True
        )
        if hasattr(model, "kpt_shape"):
            kpt_shape = model.kpt_shape  # pose-only
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, "module") else model.names  # get class names
        fp16 = True
        model.half() if fp16 else model.float()
        ch = model.yaml.get("channels", 3)
        # self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        return model
        
    except Exception as e:
        LOGGER.error(f"❌ Failed to load YOLOv8 model: {e}")
        # Fallback to direct loading
        try:
            ckpt = torch.load(weight, map_location=device)
            model = ckpt['model'] if 'model' in ckpt else ckpt
            
            # Handle compatibility issues
            for m in model.modules():
                if type(m) is nn.Upsample:
                    m.recompute_scale_factor = None
                    
            model.float()
            model.eval()
            model = model.to(device)
            
            # Try to fuse if available
            if hasattr(model, 'fuse'):
                model.fuse()
                
            return model
            
        except Exception as e2:
            LOGGER.error(f"❌ Failed to load model with fallback: {e2}")
            raise e2

def create_yolov8_dataloader(data_path, imgsz=640, batch_size=10, augment=False, 
                            hyp=None, rect=False, stride=32, workers=8, prefix=""):
    """Create YOLOv8 compatible dataloader"""

    try:
        # Create dataset
        data_dict = {
                'train': data_path,
                'val': data_path,
                'nc': 80,  # COCO classes
                'names': [f'class{i}' for i in range(80)],
                'channels': 3  # RGB channels
            }
        
        
        dataset = YOLODataset(
            img_path=data_path,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache=False,
            single_cls=False,
            stride=stride,
            pad=0.0 if augment else 0.5,
            prefix=prefix,
            task='detect',
            classes=None,
            data=data_dict,
            fraction=1.0
        )
        
        # Create dataloader
        loader = build_dataloader(
            dataset=dataset,
            batch=batch_size,
            workers=workers,
            shuffle=augment,
            rank=-1
        )
        
        return loader
        
    except Exception as e:
        LOGGER.error(f"Failed to create dataloader: {e}")
        raise e

def get_default_yolov8_hyp():
    """Get default YOLOv8 hyperparameters"""
    return {
        # Optimizer hyperparameters
        'lr0': 0.01,  # initial learning rate
        'lrf': 0.01,  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay 5e-4
        'warmup_epochs': 3.0,  # warmup epochs (fractions ok)
        'warmup_momentum': 0.8,  # warmup initial momentum
        'warmup_bias_lr': 0.1,  # warmup initial bias lr
        'box': 7.5,  # box loss gain
        'cls': 0.5,  # cls loss gain
        'dfl': 1.5,  # dfl loss gain
        'pose': 12.0,  # pose loss gain
        'kobj': 1.0,  # keypoint obj loss gain
        'label_smoothing': 0.0,  # label smoothing (fraction)
        'nbs': 64,  # nominal batch size
        'overlap_mask': True,  # masks should overlap during training
        'mask_ratio': 4,  # mask downsample ratio
        'dropout': 0.0,  # use dropout regularization
        
        # Augmentation hyperparameters
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.1,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 1.0,  # image mosaic (probability)
        'mixup': 0.0,  # image mixup (probability)
        'copy_paste': 0.0,  # segment copy-paste (probability)
        'fuse_socre': True
    }

def get_ultralytics_default_hyp():
    """Get hyperparameters directly from Ultralytics"""
    try:
        from ultralytics.utils import DEFAULT_CFG
        return DEFAULT_CFG
    except ImportError:
        try:
            from ultralytics import YOLO
            # Load a model to get default config
            model = YOLO('yolov8n.yaml')
            return model.overrides
        except:
            # Final fallback
            return get_default_yolov8_hyp()

def create_coco_train_dataloader_yolov8(cocodir, batch_size=10):
    """Create COCO training dataloader for YOLOv8"""
    train_path = f"{cocodir}/train2017.txt"

    # Load hyperparameters if available
    # hyp = None
    # hyp_files = ["data/hyps/hyp.scratch-low.yaml", "hyp.scratch-low.yaml"]
    # for hyp_file in hyp_files:
    #     if os.path.exists(hyp_file):
    #         with open(hyp_file) as f:
    #             hyp = yaml.load(f, Loader=yaml.SafeLoader)
    #         break
    
    hyp = get_ultralytics_default_hyp()

    return create_yolov8_dataloader(
        train_path, imgsz=640, batch_size=batch_size, 
        augment=True, hyp=hyp, rect=False, stride=32,
        prefix=colorstr("train: ")
    )

def create_coco_val_dataloader_yolov8(cocodir, batch_size=10, keep_images=None):
    """Create COCO validation dataloader for YOLOv8"""
    val_path = f"{cocodir}/images/val2017"
    
    hyp = get_ultralytics_default_hyp()
    
    loader = create_yolov8_dataloader(
        val_path, imgsz=640, batch_size=batch_size,
        augment=False, hyp=hyp, rect=True, stride=32,
        prefix=colorstr("val: ")
    )
    
    # Limit dataset size if requested
    if keep_images is not None:
        original_len = loader.dataset.__len__
        def limited_len():
            return min(keep_images, original_len())
        loader.dataset.__len__ = limited_len
    
    return loader

def extract_images_from_batch(batch, device):
    """Extract image tensor from YOLOv8 batch format"""
    if isinstance(batch, dict):
        # YOLOv8 Ultralytics format
        possible_keys = ['img', 'image', 'images', 'input', 'data']
        for key in possible_keys:
            if key in batch and isinstance(batch[key], torch.Tensor):
                imgs = batch[key]
                break
        else:
            # Find any 4D tensor (BCHW)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                    imgs = value
                    break
            else:
                raise ValueError(f"Could not find image tensor in batch keys: {list(batch.keys())}")
    elif isinstance(batch, (list, tuple)):
        # Traditional format [images, targets]
        imgs = batch[0]
    else:
        # Direct tensor
        imgs = batch
    
    # Ensure tensor and move to device
    if not isinstance(imgs, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(imgs)}")
    
    imgs = imgs.to(device, non_blocking=True)
    
    # Normalize if needed
    if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
        imgs = imgs.float() / 255.0
    
    return imgs

def evaluate_coco_yolov8(model, dataloader, using_cocotools=False, save_dir=".", 
                        conf_thres=0.001, iou_thres=0.65):
    """Evaluate model on COCO dataset using YOLOv8 validation"""
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    try:
        # Use Ultralytics validation if available
        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.utils import DEFAULT_CFG
        
        # Create args object with proper configuration
        args = deepcopy(DEFAULT_CFG)
        args.data = "ultralytics/cfg/datasets/coco.yaml"
        args.mode = "val"
        
        # args.conf = conf_thres
        # args.iou = iou_thres
        # args.save_json = using_cocotools
        # args.plots = True
        # args.verbose = True
        # args.half = False
        # args.task = 'detect'
        
        # Create validator with args parameter
        validator = DetectionValidator(
            dataloader=dataloader,
            save_dir=Path(save_dir),
            args=args
        )
        
        # Run validation
        results = validator(model=model)

        map = results['metrics/mAP50-95(B)']
        
        return map
        # Extract mAP@0.5:0.95
        # if hasattr(results, 'box') and hasattr(results.box, 'map'):
        #     return results.box.map
        # elif isinstance(results, dict) and 'metrics/mAP50-95' in results:
        #     return results['metrics/mAP50-95']
        # else:
        #     return 0.0
            
    except Exception as e:
        LOGGER.warning(f"Ultralytics validation failed: {e}, using fallback")
        

def export_onnx_yolov8(model, file, size=640, dynamic_batch=False, noanchor=False):
    """Export YOLOv8 model to ONNX format"""
    device = next(model.parameters()).device
    model.float()
    model.eval()

    dummy = torch.zeros(1, 3, size, size, device=device)
    
    # Prepare model for export
    for m in model.modules():
        if hasattr(m, 'export'):
            m.export = True
        if hasattr(m, 'format'):
            m.format = 'onnx'
    
    # Export using quantize module
    if noanchor:
        output_names = ["output0", "output1", "output2"]
        dynamic_axes = {
            "images": {0: "batch"}, 
            "output0": {0: "batch"}, 
            "output1": {0: "batch"}, 
            "output2": {0: "batch"}
        } if dynamic_batch else None
    else:
        output_names = ["output0"]
        dynamic_axes = {
            "images": {0: "batch"}, 
            "output0": {0: "batch"}
        } if dynamic_batch else None
    
    quantize.export_onnx(
        model, dummy, file, 
        opset_version=13,
        input_names=["images"], 
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    # Reset export flags
    for m in model.modules():
        if hasattr(m, 'export'):
            m.export = False


def cmd_quantize_yolov8(weight, cocodir, device, ignore_policy, save_ptq, save_qat, 
                       supervision_stride, iters, eval_origin, eval_ptq):
    """Main quantization command for YOLOv8"""
    quantize.initialize()

    # Create directories
    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    
    device = torch.device(device)
    model = load_yolov8_model(weight, device)

    # Verify model type
    if not isinstance(model, DetectionModel):
        LOGGER.warning(f"Model type {type(model)} may not be fully compatible with YOLOv8 QAT")

    train_dataloader = create_coco_train_dataloader_yolov8(cocodir)
    val_dataloader = create_coco_val_dataloader_yolov8(cocodir)
    
    # Apply quantization
    quantize.replace_custom_module_forward_yolov8(model)
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)
    quantize.apply_custom_rules_to_quantizer_yolov8(model, export_onnx_yolov8)
    quantize.calibrate_model(model, train_dataloader, device)
        
    # Setup summary
    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)    

    # Evaluate original model
    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])
            print(f"Original mAP: {ap:.5f}")

    # Evaluate PTQ model
    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
        summary.append(["PTQ", ap])
        print(f"PTQ mAP: {ap:.5f}")

    # Save PTQ model
    if save_ptq:
        print(f"Save PTQ model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    # QAT Fine-tuning
    best_ap = 0
    def per_epoch_callback(model, epoch, lr):
        nonlocal best_ap
        ap = evaluate_coco_yolov8(model, val_dataloader, True, json_save_dir)
        summary.append([f"QAT{epoch}", ap])
        print(f"Epoch {epoch}, mAP: {ap:.5f}")

        if ap > best_ap:
            print(f"Save QAT model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)
        
        return False  # Continue training

    def preprocess(datas):
        """Preprocess function that handles YOLOv8 dataloader format"""
        # Use the extract_images_from_batch function we already have
        try:
            return extract_images_from_batch(datas, device)
        except Exception as e:
            print(f"Error in preprocess: {e}")
            # Fallback logic
            if isinstance(datas, dict):
                # YOLOv8 format: try common keys
                for key in ['img', 'image', 'images']:
                    if key in datas and isinstance(datas[key], torch.Tensor):
                        imgs = datas[key]
                        break
                else:
                    # Find any 4D tensor
                    for key, value in datas.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) == 4:
                            imgs = value
                            break
                    else:
                        raise ValueError(f"Could not find image tensor in batch keys: {list(datas.keys())}")
            elif isinstance(datas, (list, tuple)):
                imgs = datas[0]
            else:
                imgs = datas
            
            # Process tensor
            if not isinstance(imgs, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(imgs)}")
            
            imgs = imgs.to(device, non_blocking=True)
            
            # Normalize if needed
            if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
                imgs = imgs.float() / 255.0
            
            return imgs

    def supervision_policy():
        """Create supervision policy for YOLOv8"""
        supervision_list = []
        
        # Get all modules
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList):
                supervision_list.append((name, id(module)))

        # Select modules based on stride
        keep_modules = []
        for i in range(0, len(supervision_list), supervision_stride):
            keep_modules.append(supervision_list[i][1])
        
        # Add final detection layers
        for name, module in model.named_modules():
            if 'detect' in name.lower() or 'head' in name.lower():
                keep_modules.append(id(module))

        def impl(name, module):
            module_id = id(module)
            should_supervise = module_id in keep_modules
            if should_supervise:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            return should_supervise
        
        return impl

    print("Starting QAT fine-tuning...")
    quantize.finetune(
        model, train_dataloader, per_epoch_callback, 
        early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, 
        supervision_policy=supervision_policy(),
        nepochs=10,
        prefix=colorstr("YOLOv8 ")
    )

def cmd_export_yolov8(weight, save, size, dynamic, noanchor, noqadd):
    """Export YOLOv8 quantized model to ONNX"""
    quantize.initialize()
    
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
        
    try:
        model = torch.load(weight, map_location="cpu")["model"]
    except:
        model = load_yolov8_model(weight, "cpu")
    
    if not noqadd:
        quantize.replace_custom_module_forward_yolov8(model)

    export_onnx_yolov8(model, save, size, dynamic_batch=dynamic, noanchor=noanchor)
    print(f"Save ONNX to {save}")

def cmd_sensitive_analysis_yolov8(weight, device, cocodir, summary_save, num_image):
    """Sensitive layer analysis for YOLOv8"""
    quantize.initialize()
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    
    train_dataloader = create_coco_train_dataloader_yolov8(cocodir)
    val_dataloader = create_coco_val_dataloader_yolov8(
        cocodir, keep_images=None if num_image is None or num_image < 1 else num_image
    )
    
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco_yolov8(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    layer_names = []
    for name, module in model.named_modules():
        if quantize.have_quantizer(module):
            layer_names.append((name, module))

    for name, layer in layer_names:
        print(f"Quantization disable {name}")
        quantize.disable_quantization(layer).apply()
        ap = evaluate_coco_yolov8(model, val_dataloader)
        summary.append([ap, name])
        quantize.enable_quantization(layer).apply()
    
    summary_sorted = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary_sorted[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")

def cmd_test_yolov8(weight, device, cocodir, confidence, nmsthres):
    """Test YOLOv8 model"""
    device = torch.device(device)
    model = load_yolov8_model(weight, device)
    val_dataloader = create_coco_val_dataloader_yolov8(cocodir)
    
    ap = evaluate_coco_yolov8(
        model, val_dataloader, True, 
        conf_thres=confidence, iou_thres=nmsthres
    )
    print(f"Test mAP: {ap:.5f}")

def init_seeds(seed=0):
    """Initialize random seeds"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='qat_yolov8.py')
    subps = parser.add_subparsers(dest="cmd")
    
    # Export command
    exp = subps.add_parser("export", help="Export weight to ONNX file")
    exp.add_argument("weight", type=str, default="yolov8n.pt", help="export pt file")
    exp.add_argument("--save", type=str, required=False, help="export onnx file")
    exp.add_argument("--size", type=int, default=640, help="export input size")
    exp.add_argument("--dynamic", action="store_true", help="export dynamic batch")
    exp.add_argument("--noanchor", action="store_true", help="export no anchor nodes")
    exp.add_argument("--noqadd", action="store_true", help="export do not add QuantAdd")

    # Quantize command
    qat = subps.add_parser("quantize", help="PTQ/QAT finetune for YOLOv8")
    qat.add_argument("weight", type=str, nargs="?", default="yolov8n.pt", help="weight file")
    qat.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument("--ignore-policy", type=str, default=r"model\.22\.dfl\.(.*)", help="regex for layers to ignore")
    qat.add_argument("--ptq", type=str, default="ptq_yolov8.pt", help="PTQ model save file")
    qat.add_argument("--qat", type=str, default=None, help="QAT model save file")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--iters", type=int, default=200, help="iterations per epoch")
    qat.add_argument("--eval-origin", action="store_true", help="evaluate original model")
    qat.add_argument("--eval-ptq", action="store_true", help="evaluate PTQ model")

    # Sensitive analysis command
    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument("weight", type=str, nargs="?", default="yolov8n.pt", help="weight file")
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")
    sensitive.add_argument("--num-image", type=int, default=None, help="number of images to evaluate")

    # Test command
    testcmd = subps.add_parser("test", help="Test model evaluation")
    testcmd.add_argument("weight", type=str, default="yolov8n.pt", help="weight file")
    testcmd.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--nmsthres", type=float, default=0.65, help="NMS threshold")

    args = parser.parse_args()
    init_seeds(57)

    if args.cmd == "export":
        cmd_export_yolov8(args.weight, args.save, args.size, args.dynamic, args.noanchor, args.noqadd)
    elif args.cmd == "quantize":
        print(args)
        cmd_quantize_yolov8(
            args.weight, args.cocodir, args.device, args.ignore_policy, 
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq
        )
    elif args.cmd == "sensitive":
        cmd_sensitive_analysis_yolov8(args.weight, args.device, args.cocodir, args.summary, args.num_image)
    elif args.cmd == "test":
        cmd_test_yolov8(args.weight, args.device, args.cocodir, args.confidence, args.nmsthres)
    else:
        parser.print_help()