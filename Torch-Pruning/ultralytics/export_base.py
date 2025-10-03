#!/usr/bin/env python3
# export_pruned_onnx_trt.py
import argparse
from ultralytics import YOLO

def export_model(weights, topk_all, iou_thres, conf_thres, class_agnostic, pooler_scale, sampling_ratio, mask_resolution):
    yolo_model = YOLO(weights)  

    # Finally export to onnx_trt
    yolo_model.export(
        format="onnx_trt",
        dynamic=True,
        topk_all=topk_all,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
        class_agnostic=class_agnostic,
        pooler_scale=pooler_scale,
        sampling_ratio=sampling_ratio,
        mask_resolution=mask_resolution
    )


def main():
    parser = argparse.ArgumentParser(description="Export pruned YOLO model to ONNX with TensorRT optimization.")
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the YOLO weights file (pruned .pt)')
    parser.add_argument('--topk_all', type=int, default=100)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--class_agnostic', action='store_true')
    parser.add_argument('--pooler_scale', type=float, default=0.25)
    parser.add_argument('--sampling_ratio', type=int, default=0)
    parser.add_argument('--mask_resolution', type=int, default=160)
    args = parser.parse_args()
    export_model(args.weights, args.topk_all, args.iou_thres, args.conf_thres,
                 args.class_agnostic, args.pooler_scale, args.sampling_ratio, args.mask_resolution)

if __name__ == "__main__":
    main()
