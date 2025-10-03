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
import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

# PyTorch
import torch
import torch.optim as optim
from torch.cuda import amp

# Pytorch Quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from absl import logging as quant_logging

# Custom Rules for YOLOv8
try:
    from quantization.rules import find_quantizer_pairs
except ImportError:
    # Fallback if rules module not found
    def find_quantizer_pairs(onnx_file):
        return []

class QuantAdd(torch.nn.Module):
    """Quantized Add operation for YOLOv8"""
    def __init__(self, quantization):
        super().__init__()

        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input0_quantizer._calibrator._torch_hist = True
            self._input1_quantizer._calibrator._torch_hist = True
            self._fake_quant = True
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y

class QuantConcat(torch.nn.Module):
    """Quantized Concat operation for YOLOv8"""
    def __init__(self, quantization=True):
        super().__init__()
        self.quantization = quantization
        if quantization:
            self._input_quantizers = torch.nn.ModuleList()
    
    def add_input_quantizer(self):
        if self.quantization:
            quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            quantizer._calibrator._torch_hist = True
            self._input_quantizers.append(quantizer)
            return quantizer
        return None
    
    def forward(self, inputs, dim=1):
        if self.quantization and len(self._input_quantizers) > 0:
            # Quantize each input
            quantized_inputs = []
            for i, inp in enumerate(inputs):
                if i < len(self._input_quantizers):
                    quantized_inputs.append(self._input_quantizers[i](inp))
                else:
                    quantized_inputs.append(inp)
            return torch.cat(quantized_inputs, dim=dim)
        return torch.cat(inputs, dim=dim)

class QuantUpsample(torch.nn.Module):
    """Quantized Upsample for YOLOv8"""
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, quantization=True):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.quantization = quantization
        
        # Add YOLOv8/v9 compatibility attributes
        self.f = -1  # Default flow index
        self.i = -1  # Default layer index
        self.type = 'QuantUpsample'  # Layer type identifier
        
        if quantization:
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input_quantizer._calibrator._torch_hist = True
    
    def forward(self, x):
        if self.quantization:
            x = self._input_quantizer(x)
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                                             mode=self.mode, align_corners=self.align_corners)

class disable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)

class enable_quantization:
    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)

def have_quantizer(module):
    for name, submodule in module.named_modules():
        if isinstance(submodule, quant_nn.TensorQuantizer):
            return True
    return False

# Initialize PyTorch Quantization for YOLOv8
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)

def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance

def quantization_ignore_match(ignore_policy: Union[str, List[str], Callable], path: str) -> bool:
    if ignore_policy is None: 
        return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):
        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: 
            return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False

# YOLOv8 Bottleneck quantization forward
def bottleneck_quant_forward_yolov8(self, x):
    """Modified forward for YOLOv8 Bottleneck with quantization"""
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# YOLOv8 C2f quantization forward
def c2f_quant_forward_yolov8(self, x):
    """Modified forward for YOLOv8 C2f with quantization"""
    # y = list(self.cv1(x).split((self.c, self.c), 1))
    y = [self.cv0(x), self.cv1(x)]
    for m in self.m:
        y.append(m(y[-1]))
    
    # Use quantized concat if available
    if hasattr(self, "concat_op"):
        return self.cv2(self.concat_op(y, 1))
    return self.cv2(torch.cat(y, 1))

# Replace Bottleneck forward for YOLOv8
def replace_bottleneck_forward_yolov8(model):
    """Replace Bottleneck forward method with quantized version for YOLOv8"""
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if hasattr(bottleneck, 'add') and bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    bottleneck.addop = QuantAdd(True)
                bottleneck.__class__.forward = bottleneck_quant_forward_yolov8

# Replace C2f forward for YOLOv8  
def replace_c2f_forward_yolov8(model):
    """Replace C2f forward method with quantized version for YOLOv8"""
    for name, c2f in model.named_modules():
        if c2f.__class__.__name__ == "C2f":
            if not hasattr(c2f, "concat_op"):
                print(f"Add QuantConcat to {name}")
                c2f.concat_op = QuantConcat(True)
            c2f.__class__.forward = c2f_quant_forward_yolov8

# Replace custom modules forward
def replace_custom_module_forward_yolov8(model):
    """Replace custom module forwards for YOLOv8"""
    replace_bottleneck_forward_yolov8(model)
    replace_c2f_forward_yolov8(model)
    
    # Replace Upsample
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Upsample):
            # Replace with quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                quant_upsample = QuantUpsample(
                    size=module.size,
                    scale_factor=module.scale_factor, 
                    mode=module.mode,
                    align_corners=module.align_corners,
                    quantization=True
                )
                quant_upsample.i = module.i
                quant_upsample.f = module.f
                
                setattr(parent, child_name, quant_upsample)
                print(f"Replace Upsample with QuantUpsample at {name}")

def replace_to_quantization_module(model: torch.nn.Module, ignore_policy: Union[str, List[str], Callable] = None):
    """Replace standard modules with quantized versions"""
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue
                    
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)

def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))

def apply_custom_rules_to_quantizer_yolov8(model: torch.nn.Module, export_onnx: Callable):
    """Apply custom quantization rules for YOLOv8"""
    # Apply rules to graph
    try:
        export_onnx(model, "quantization-custom-rules-temp.onnx")
        pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
        for major, sub in pairs:
            print(f"Rules: {sub} match to {major}")
            try:
                get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
            except:
                print(f"Warning: Could not apply rule {sub} -> {major}")
        if os.path.exists("quantization-custom-rules-temp.onnx"):
            os.remove("quantization-custom-rules-temp.onnx")
    except Exception as e:
        print(f"Warning: Could not apply ONNX-based rules: {e}")

    # Apply YOLOv8 specific rules for Bottleneck
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if hasattr(bottleneck, 'add') and bottleneck.add and hasattr(bottleneck, 'addop'):
                try:
                    print(f"Rules: {name}.add match to {name}.cv1")
                    if hasattr(bottleneck.cv1, 'conv'):
                        major = bottleneck.cv1.conv._input_quantizer
                    else:
                        major = bottleneck.cv1._input_quantizer
                    bottleneck.addop._input0_quantizer = major
                    bottleneck.addop._input1_quantizer = major
                except Exception as e:
                    print(f"Warning: Could not apply Bottleneck rule for {name}: {e}")

    # Apply rules for C2f concat operations
    for name, c2f in model.named_modules():
        if c2f.__class__.__name__ == "C2f" and hasattr(c2f, 'concat_op'):
            try:
                # Match concat quantizers to cv1 input quantizer
                if hasattr(c2f.cv1, 'conv'):
                    major = c2f.cv1.conv._input_quantizer
                elif hasattr(c2f.cv1, '_input_quantizer'):
                    major = c2f.cv1._input_quantizer
                else:
                    continue
                    
                # Add quantizers for each input to concat
                for i in range(2 + len(c2f.m)):  # 2 from split + bottlenecks
                    c2f.concat_op.add_input_quantizer()
                
                # Set all input quantizers to match major
                for quantizer in c2f.concat_op._input_quantizers:
                    quantizer = major
                print(f"Rules: {name}.concat inputs match to {name}.cv1")
            except Exception as e:
                print(f"Warning: Could not apply C2f rule for {name}: {e}")

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

def calibrate_model(model: torch.nn.Module, dataloader, device, num_batch=25):
    """Calibrate quantized model"""
    def compute_amax(model, device, **kwargs):  # Add device parameter
        successful_calibrations = 0
        failed_calibrations = 0
        
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    # try:
                        # Check calibrator type and call appropriate method
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            # MaxCalibrator doesn't support 'method' parameter
                            module.load_calib_amax(strict=False)
                        elif isinstance(module._calibrator, calib.HistogramCalibrator):
                            # HistogramCalibrator supports 'method' parameter
                            module.load_calib_amax(strict=False, **kwargs)
                        else:
                            # Try without method parameter first
                            try:
                                module.load_calib_amax(strict=False)
                            except TypeError:
                                # If it fails, try with method parameter
                                module.load_calib_amax(strict=False, **kwargs)

                        if module._amax is not None:
                            module._amax = module._amax.to(device)
                            successful_calibrations += 1
                        else:
                            failed_calibrations += 1
                            
                    # except Exception as e:
                    #     print(f"Warning: Failed to load amax for {name}: {e}")
                    #     failed_calibrations += 1
        
        print(f"Calibration results: {successful_calibrations} successful, {failed_calibrations} failed")
    
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        active_quantizers = []
        
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                    active_quantizers.append(name)
                else:
                    module.disable()

        print(f"Found {len(active_quantizers)} active quantizers: {active_quantizers[:5]}{'...' if len(active_quantizers) > 5 else ''}")

        # Feed data to the network for collecting stats
        processed_batches = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                # try:
                imgs = extract_images_from_batch(batch, device)
                
                # Ensure proper preprocessing
                if imgs.dtype == torch.uint8:
                    imgs = imgs.float() / 255.0
                elif imgs.max() > 1.0:
                    imgs = imgs / 255.0
                
                # Forward pass
                _ = model(imgs)
                processed_batches += 1
                    
                # except Exception as e:
                #     print(f"Error processing batch {i}: {e}")
                #     continue

                if i >= num_batch:
                    break

        print(f"Successfully processed {processed_batches}/{min(num_batch, len(data_loader))} batches")

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, device, method="mse")  # Pass device parameter

def finetune(
    model: torch.nn.Module, train_dataloader, per_epoch_callback: Callable = None, preprocess: Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule: Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy: Callable = None, prefix=""
):
    """QAT Fine-tuning for YOLOv8"""
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()
    model.train()
    for param in model.parameters():
        param.requires_grad = True
        
    origin_model.train()
    for param in origin_model.parameters():
        param.requires_grad = False
    # origin_model.requires_grad_(False)

    scaler = amp.GradScaler(enabled=fp16)
    optimizer = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            # Handle tuple/list outputs by taking first element or flattening
            if isinstance(output, (tuple, list)):
                # For YOLOv8, usually take the first tensor output
                for item in output:
                    if isinstance(item, torch.Tensor):
                        l.append(item)
                        break
                else:
                    # If no tensor found, take first item
                    l.append(output[0] if len(output) > 0 else output)
            else:
                l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): 
            continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])

    for iepoch in range(nepochs):
        model.train()
        for param in model.parameters():
            param.requires_grad = True
        
        origin_model.train()
        for param in origin_model.parameters():
            param.requires_grad = False
            
        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs = []
        origin_outputs = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc=f"{prefix}QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):
            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
            else:
                # Handle different dataloader formats
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
                
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    # Ensure both are tensors before computing loss
                    if isinstance(mo, torch.Tensor) and isinstance(fo, torch.Tensor):
                        # Check if shapes match
                        if mo.shape == fo.shape:
                            quant_loss += quant_lossfn(mo, fo)
                        else:
                            print(f"Warning: Shape mismatch at index {index}: {mo.shape} vs {fo.shape}")
                    else:
                        print(f"Warning: Non-tensor outputs at index {index}: {type(mo)} vs {type(fo)}")

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"{prefix}QAT Finetuning {iepoch + 1}/{nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # Remove hooks
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break

def export_onnx(model, input, file, *args, **kwargs):
    """Export quantized model to ONNX"""
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False

def remove_redundant_qdq_model(input_onnx_path, output_onnx_path):
    """Remove redundant Q/DQ nodes from ONNX model"""
    try:
        import onnx
        from onnxsim import simplify
        
        # Load model
        model = onnx.load(input_onnx_path)
        
        # Simplify model to remove redundant Q/DQ
        model_simplified, check = simplify(model)
        
        if check:
            onnx.save(model_simplified, output_onnx_path)
            print(f"Simplified model saved to {output_onnx_path}")
        else:
            print("Simplification failed, saving original model")
            onnx.save(model, output_onnx_path)
            
    except ImportError:
        print("onnxsim not installed, copying original file")
        import shutil
        shutil.copy(input_onnx_path, output_onnx_path)
    except Exception as e:
        print(f"Error in simplification: {e}")
        import shutil
        shutil.copy(input_onnx_path, output_onnx_path)

# Aliases for backward compatibility
replace_bottleneck_forward = replace_bottleneck_forward_yolov8
apply_custom_rules_to_quantizer = apply_custom_rules_to_quantizer_yolov8
replace_custom_module_forward = replace_custom_module_forward_yolov8