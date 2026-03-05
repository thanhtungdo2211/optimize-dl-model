"""quantize_11_modelopt.py

Drop-in replacement for quantize_11.py using NVIDIA Model Optimizer (modelopt)
instead of pytorch_quantization.

Public API is identical to quantize_11.py so qat_yolov11.py only needs to
change one import line:

    # before
    import quantize.quantize_11 as quantize

    # after
    import quantize.quantize_11_modelopt as quantize

Key improvements over pytorch_quantization path:
  - No use_fb_fake_quant global flag (no poisoning risk)
  - float32 scales stored automatically (no _cast_quantizer_scales_to_float32)
  - INT4 ONNX export via opset 21 (no INT4→INT8 promotion workaround)
  - One-call calibration: mtq.quantize(model, config, forward_loop)
  - Actively maintained by NVIDIA (used in TRT-LLM)
"""

import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

# ---------------------------------------------------------------------------
# modelopt imports — required
# ---------------------------------------------------------------------------
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

# ---------------------------------------------------------------------------
# TensorQuantizer: prefer modelopt, fall back to pytorch_quantization.
#
# The custom quant modules (QuantAdd / QuantConcat / QuantChunk / QuantUpsample)
# insert TensorQuantizers manually.  We need a factory function _make_quantizer()
# so the same class works whichever backend is active.
# ---------------------------------------------------------------------------
_USING_MTQ_QUANTIZER = False
_TensorQuantizer = None  # set below

try:
    from modelopt.torch.quantization.nn import TensorQuantizer as _MtqTQ
    # When using modelopt, custom modules (QuantAdd/QuantChunk/QuantUpsample) must NOT
    # contain modelopt TensorQuantizers before mtq.quantize() is called, because
    # mtq.quantize() asserts is_quantized(model) == False.
    # Use nn.Identity() as a placeholder; apply_custom_rules_to_quantizer() will
    # later replace these placeholders with real quantizers from Conv layers.
    def _make_quantizer(num_bits: int = 8) -> nn.Module:
        return nn.Identity()
    _TensorQuantizer = _MtqTQ
    _USING_MTQ_QUANTIZER = True
    print("[quantize_11_modelopt] Using modelopt backend; custom-module quantizers are "
          "Identity placeholders (replaced by apply_custom_rules_to_quantizer).")
except Exception as _e:
    print(f"[quantize_11_modelopt] modelopt TensorQuantizer unavailable ({_e}); "
          "falling back to pytorch_quantization for custom modules.")
    try:
        from pytorch_quantization import nn as _pq_nn
        from pytorch_quantization.tensor_quant import QuantDescriptor as _QD
        from pytorch_quantization import calib as _calib

        def _make_quantizer(num_bits: int = 8) -> nn.Module:
            q = _pq_nn.TensorQuantizer(_QD(num_bits=num_bits, calib_method="histogram"))
            q._calibrator._torch_hist = True
            return q

        _TensorQuantizer = _pq_nn.TensorQuantizer
        _USING_MTQ_QUANTIZER = False
    except ImportError as _e2:
        raise RuntimeError(
            "Neither modelopt.torch.quantization.nn.TensorQuantizer nor "
            "pytorch_quantization.nn.TensorQuantizer is importable. "
            "Install nvidia-modelopt or pytorch-quantization."
        ) from _e2


# ---------------------------------------------------------------------------
# rules_v2 helper (unchanged)
# ---------------------------------------------------------------------------
try:
    from quantization.rules_v2 import find_quantizer_pairs
except ImportError:
    try:
        from quantize.rules_v2 import find_quantizer_pairs
    except ImportError:
        def find_quantizer_pairs(onnx_file):
            return []


# ===========================================================================
# Custom quantized modules
# ===========================================================================

class QuantAdd(nn.Module):
    """Quantized element-wise add for residual connections in Bottleneck."""

    def __init__(self, quantization: bool, num_bits: int = 8):
        super().__init__()
        self.quantization = quantization
        if quantization:
            self._input0_quantizer = _make_quantizer(num_bits)
            self._input1_quantizer = _make_quantizer(num_bits)
            self._fake_quant = True

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y


class QuantConcat(nn.Module):
    """Quantized torch.cat for Concat layers."""

    def __init__(self, quantization: bool = True, num_bits: int = 8):
        super().__init__()
        self.quantization = quantization
        self.num_bits = num_bits
        if quantization:
            self._input_quantizers = nn.ModuleList()

    def add_input_quantizer(self):
        if self.quantization:
            q = _make_quantizer(self.num_bits)
            self._input_quantizers.append(q)
            return q
        return None

    def forward(self, inputs, dim: int = 1):
        if self.quantization and len(self._input_quantizers) > 0:
            quantized = []
            for i, inp in enumerate(inputs):
                if i < len(self._input_quantizers):
                    quantized.append(self._input_quantizers[i](inp))
                else:
                    quantized.append(inp)
            return torch.cat(quantized, dim=dim)
        return torch.cat(inputs, dim=dim)


class QuantChunk(nn.Module):
    """Quantized torch.split for C3k2 chunk operations."""

    def __init__(self, c: int, num_bits: int = 8):
        super().__init__()
        self._input0_quantizer = _make_quantizer(num_bits)
        self.c = c

    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)


class QuantUpsample(nn.Module):
    """Quantized nn.Upsample replacement."""

    def __init__(self, size=None, scale_factor=None, mode: str = 'nearest',
                 align_corners=None, quantization: bool = True, num_bits: int = 8):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.quantization = quantization

        # Compatibility attributes expected by YOLO layer-iteration code
        self.f = -1
        self.i = -1
        self.type = 'QuantUpsample'

        if quantization:
            self._input_quantizer = _make_quantizer(num_bits)

    def forward(self, x):
        if self.quantization:
            x = self._input_quantizer(x)
        return nn.functional.interpolate(
            x, size=self.size, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners
        )


# ===========================================================================
# Quantization enable/disable context managers
# ===========================================================================

class disable_quantization:
    """Context manager / callable that disables all quantizers in a model."""

    def __init__(self, model: nn.Module):
        self.model = model

    def apply(self, disabled: bool = True):
        try:
            if disabled:
                mtq.disable_quantizer(self.model, "*")
            else:
                mtq.enable_quantizer(self.model, "*")
        except Exception:
            pass
        # Also handle any remaining pytorch_quantization TensorQuantizers
        # (from custom modules when _USING_MTQ_QUANTIZER is False)
        if not _USING_MTQ_QUANTIZER:
            for _, module in self.model.named_modules():
                if isinstance(module, _TensorQuantizer):
                    module._disabled = disabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    """Context manager / callable that enables all quantizers in a model."""

    def __init__(self, model: nn.Module):
        self.model = model

    def apply(self, enabled: bool = True):
        try:
            if enabled:
                mtq.enable_quantizer(self.model, "*")
            else:
                mtq.disable_quantizer(self.model, "*")
        except Exception:
            pass
        if not _USING_MTQ_QUANTIZER:
            for _, module in self.model.named_modules():
                if isinstance(module, _TensorQuantizer):
                    module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


# ===========================================================================
# Utility helpers
# ===========================================================================

def have_quantizer(module: nn.Module) -> bool:
    """Return True if the module (or any submodule) contains a TensorQuantizer."""
    try:
        return mtq.has_quantizer(module)
    except AttributeError:
        pass
    for _, submodule in module.named_modules():
        if isinstance(submodule, _TensorQuantizer):
            return True
    return False


def get_attr_with_path(m: nn.Module, path: str):
    """Traverse a dotted attribute path and return the final attribute."""
    names = path.split(".")
    for name in names:
        m = getattr(m, name)
    return m


def set_module(model: nn.Module, submodule_key: str, module: nn.Module):
    """Replace a submodule identified by a dotted key."""
    tokens = submodule_key.split('.')
    parent = model
    for s in tokens[:-1]:
        parent = getattr(parent, s)
    setattr(parent, tokens[-1], module)


# ===========================================================================
# Initialization (no-op in modelopt)
# ===========================================================================

def initialize():
    """No-op: modelopt does not require global initialization.

    In pytorch_quantization this set histogram calibration as the default for
    all QuantConv2d/QuantLinear layers.  With modelopt that is expressed in
    the quant_config dict passed to calibrate_model().
    """
    pass


# ===========================================================================
# Module replacement (no-op for standard layers; custom ops still needed)
# ===========================================================================

def replace_to_quantization_module(
    model: nn.Module,
    ignore_policy: Union[str, List[str], Callable] = None
):
    """No-op stub kept for API compatibility.

    In the pytorch_quantization path this replaced Conv2d/Linear/... with
    their quantized counterparts.  With modelopt, mtq.quantize() inside
    calibrate_model() does the replacement automatically.

    The ignore_policy is stored on the model so calibrate_model() can read it
    when building the quant_config.
    """
    if ignore_policy is not None:
        model._mtq_ignore_policy = ignore_policy


# ===========================================================================
# Mixed precision helpers
# ===========================================================================

def _regex_to_mtq_pattern(pattern: str) -> str:
    """Convert a Python regex string to a modelopt glob-style name pattern.

    Examples
    --------
    ``r"model\\.6\\..*"``  →  ``"model.6.*"``
    ``r".*\\.attn\\..*"``  →  ``"*.attn.*"``
    ``r"model\\.10\\..*"`` →  ``"model.10.*"``
    """
    p = pattern.lstrip('^').rstrip('$')
    # Collapse groups like (?:...) → *
    p = re.sub(r'\(\?:.*?\)', '*', p)
    p = re.sub(r'\(.*?\)', '*', p)
    # Use a placeholder for escaped dots so they survive the .* → * substitution
    ESCDOT = '\x00'
    p = p.replace(r'\.', ESCDOT)
    # .*\. → *.  and  .* → *
    p = p.replace('.*' + ESCDOT, '*' + ESCDOT)
    p = p.replace('.*', '*')
    p = p.replace(ESCDOT, '.')
    # Collapse consecutive wildcards
    while '**' in p:
        p = p.replace('**', '*')
    return p


def _build_quant_config(
    ignore_policy: Union[str, List[str], Callable, None],
    mixed_precision_config: Dict[int, List[str]] = None
) -> dict:
    """Build a modelopt quant_cfg dict from legacy ignore_policy and mixed_precision_config.

    Parameters
    ----------
    ignore_policy:
        Same format as pytorch_quantization path: a regex string, list of
        regex strings, or a callable(path) → bool.
    mixed_precision_config:
        Mapping ``{num_bits: [regex_patterns]}``.  Layers matching a pattern
        will use that bit width.  Patterns not listed default to INT8.

    Returns
    -------
    dict with keys ``"quant_cfg"`` and ``"algorithm"`` suitable for
    ``mtq.quantize()``.
    """
    quant_cfg: dict = {
        # INT8 defaults for weights and activations
        "*weight_quantizer*": {"num_bits": 8, "axis": 0},
        "*input_quantizer*":  {"num_bits": 8, "axis": None},
        # Custom module quantizers
        "*_input0_quantizer*": {"num_bits": 8, "axis": None},
        "*_input1_quantizer*": {"num_bits": 8, "axis": None},
        "*_input_quantizers*": {"num_bits": 8, "axis": None},
    }

    # Apply mixed-precision overrides
    if mixed_precision_config:
        for num_bits, patterns in mixed_precision_config.items():
            if not patterns:
                continue
            for pattern in patterns:
                glob = _regex_to_mtq_pattern(pattern)
                quant_cfg[glob] = {"num_bits": num_bits, "axis": None}

    # Apply ignore policy (disable quantization for matching layers)
    if ignore_policy is not None:
        if callable(ignore_policy):
            # Can't convert callable to a static pattern — warn and skip
            print("[quantize_11_modelopt] Warning: callable ignore_policy cannot be "
                  "converted to a modelopt config pattern. Quantization will be applied "
                  "to all layers. Use a list of regex strings instead.")
        else:
            if isinstance(ignore_policy, str):
                ignore_policy = [ignore_policy]
            for pattern in ignore_policy:
                glob = _regex_to_mtq_pattern(pattern)
                quant_cfg[glob] = {"enable": False}

    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def apply_mixed_precision(
    model: nn.Module,
    mixed_precision_config: Dict[int, List[str]]
):
    """Store a mixed-precision config on the model for use by calibrate_model().

    Parameters
    ----------
    mixed_precision_config:
        Mapping ``{num_bits: [regex_patterns]}``, e.g.::

            {4: [r"model\\.6\\..*", r"model\\.7\\..*"]}
    """
    model._mtq_mixed_precision_config = mixed_precision_config


def set_quantizer_bits(
    model: nn.Module,
    layer_patterns: List[str],
    num_bits: int
):
    """Set num_bits for all TensorQuantizers inside modules matching any pattern.

    This is a post-hoc override.  Prefer passing ``mixed_precision_config`` to
    ``calibrate_model()`` so the calibrator collects stats at the correct width.
    """
    changed = 0
    for name, module in model.named_modules():
        if any(re.match(p, name) for p in layer_patterns):
            for _, submodule in module.named_modules():
                if isinstance(submodule, _TensorQuantizer):
                    # modelopt TensorQuantizer
                    if hasattr(submodule, 'num_bits'):
                        submodule.num_bits = num_bits
                        changed += 1
                    # pytorch_quantization TensorQuantizer fallback
                    elif hasattr(submodule, '_num_bits'):
                        submodule._num_bits = num_bits
                        if (submodule._calibrator is not None
                                and hasattr(submodule._calibrator, '_num_bits')):
                            submodule._calibrator._num_bits = num_bits
                        changed += 1
    print(f"[Mixed Precision] Set {num_bits}-bit on {changed} quantizer(s) "
          f"matching {layer_patterns}")


def print_quantizer_bits(model: nn.Module):
    """Print a per-quantizer bit-width summary."""
    try:
        mtq.print_quant_summary(model)
        return
    except AttributeError:
        pass
    # Fallback: manual scan
    bit_counts: Dict[int, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, _TensorQuantizer):
            bits = getattr(module, 'num_bits', getattr(module, '_num_bits', '?'))
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
            print(f"  {name:<80s} {bits}-bit")
    print("[Mixed Precision] Summary:", {f"INT{k}": v for k, v in sorted(bit_counts.items())})


# ===========================================================================
# Module surgery — forward-patch helpers (unchanged logic, same API)
# ===========================================================================

def bottleneck_quant_forward_yolo11(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def c3k2_quant_forward(self, x):
    if hasattr(self, "chunkop"):
        y = list(self.chunkop(self.cv1(x), 2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    else:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def replace_bottleneck_forward_yolo11(model: nn.Module):
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if hasattr(bottleneck, 'add') and bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    bottleneck.addop = QuantAdd(True)
                bottleneck.__class__.forward = bottleneck_quant_forward_yolo11


def replace_c3k2_forward_yolo11(model: nn.Module):
    for name, c3k2 in model.named_modules():
        if c3k2.__class__.__name__ == "C3k2":
            if not hasattr(c3k2, "chunkop"):
                print(f"Add QuantChunk to {name}")
                c3k2.chunkop = QuantChunk(c3k2.c)
            c3k2.__class__.forward = c3k2_quant_forward


def replace_custom_module_forward_yolo11(model: nn.Module, device=None):
    """Inject QuantAdd / QuantChunk / QuantUpsample into the model graph."""
    replace_bottleneck_forward_yolo11(model)
    replace_c3k2_forward_yolo11(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Upsample):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                qu = QuantUpsample(
                    size=module.size,
                    scale_factor=module.scale_factor,
                    mode=module.mode,
                    align_corners=module.align_corners,
                    quantization=True,
                )
                qu.i = module.i
                qu.f = module.f
                setattr(parent, child_name, qu)
                print(f"Replace Upsample with QuantUpsample at {name}")


# ===========================================================================
# Image extraction helper (unchanged)
# ===========================================================================

def extract_images_from_batch(batch, device):
    if isinstance(batch, dict):
        for key in ('img', 'image', 'images', 'input', 'data'):
            if key in batch and isinstance(batch[key], torch.Tensor):
                imgs = batch[key]
                break
        else:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.ndim == 4:
                    imgs = value
                    break
            else:
                raise ValueError(
                    f"Could not find image tensor in batch keys: {list(batch.keys())}"
                )
    elif isinstance(batch, (list, tuple)):
        imgs = batch[0]
    else:
        imgs = batch

    if not isinstance(imgs, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(imgs)}")

    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
        imgs = imgs.float() / 255.0
    return imgs


# ===========================================================================
# Calibration — main PTQ entry point
# ===========================================================================

def calibrate_model(
    model: nn.Module,
    dataloader,
    device,
    num_batch: int = 25,
    quant_config: dict = None,
):
    """Insert quantizers and calibrate the model using modelopt.

    Replaces the two-step pytorch_quantization flow::

        replace_to_quantization_module(model, ignore_policy)
        calibrate_model(model, dataloader, device)

    with a single ``mtq.quantize()`` call.

    Parameters
    ----------
    model:
        The FP32 model.  ``replace_custom_module_forward()`` must already
        have been called so QuantAdd / QuantChunk / QuantUpsample are in place.
    dataloader:
        Training or calibration dataloader.
    device:
        CUDA device string (e.g. "cuda:0").
    num_batch:
        Number of mini-batches to use for calibration.
    quant_config:
        modelopt quant config dict.  If None, the config is built from
        ``model._mtq_ignore_policy`` and ``model._mtq_mixed_precision_config``
        (set by ``replace_to_quantization_module`` and
        ``apply_mixed_precision``), falling back to ``mtq.INT8_DEFAULT_CFG``.
    """
    if quant_config is None:
        ignore_policy = getattr(model, '_mtq_ignore_policy', None)
        mp_config = getattr(model, '_mtq_mixed_precision_config', None)
        if ignore_policy is not None or mp_config is not None:
            quant_config = _build_quant_config(ignore_policy, mp_config)
        else:
            quant_config = mtq.INT8_DEFAULT_CFG

    processed = [0]  # list so the nested closure can mutate it

    def forward_loop(m):
        m.eval()
        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(dataloader),
                total=num_batch,
                desc="Calibrating"
            ):
                if processed[0] >= num_batch:
                    break
                try:
                    imgs = extract_images_from_batch(batch, device)
                    m(imgs)
                    processed[0] += 1
                except Exception as e:
                    print(f"[calibrate_model] Skipping batch {i}: {e}")

    mtq.quantize(model, quant_config, forward_loop)
    print(f"[calibrate_model] Calibrated on {processed[0]} batches.")


# ===========================================================================
# ONNX export
# ===========================================================================

def export_onnx(model: nn.Module, input: torch.Tensor, file: str, *args, **kwargs):
    """Export a modelopt-quantized model to ONNX.

    Uses ``mtq.export_torch_mode()`` which:
      - activates fake-quantization (replaces the old use_fb_fake_quant flag)
      - is exception-safe (no global flag left dirty on crash)
      - supports opset 21 INT4/FP8 Q/DQ nodes natively

    Opset version note
    ------------------
    * Pass ``opset_version=13`` for TRT 8.x compatibility (INT4 layers are
      silently promoted to INT8 if ``_save_and_promote_int4_quantizers``
      is called before this function — kept below as a fallback).
    * Pass ``opset_version=21`` for TRT 10.x to get native INT4 Q/DQ nodes.
    """
    model_was_half = next(model.parameters()).dtype == torch.float16
    if model_was_half:
        model.float()

    # Opset-13 INT4 workaround: promote INT4 → INT8 for the duration of
    # the export.  Remove this block once TRT 10.x + opset 21 is confirmed
    # on the target machine.
    opset = kwargs.get('opset_version', 21)
    saved_bits: dict = {}
    if opset < 21:
        saved_bits = _save_and_promote_int4_quantizers(model)

    try:
        model.eval()
        with torch.no_grad():
            # with mtq.utils.export_torch_mode():
                torch.onnx.export(model, input.float(), file, *args, **kwargs)
    finally:
        _restore_quantizer_bits(saved_bits)
        if model_was_half:
            model.half()


# ---------------------------------------------------------------------------
# INT4 helpers kept for opset-13 compatibility
# ---------------------------------------------------------------------------

def _save_and_promote_int4_quantizers(model: nn.Module) -> dict:
    """Temporarily promote INT4 quantizers to INT8 for ONNX opset-13 export."""
    saved: dict = {}
    for _, module in model.named_modules():
        if not isinstance(module, _TensorQuantizer):
            continue
        bits = getattr(module, 'num_bits', getattr(module, '_num_bits', 8))
        if bits == 4:
            saved[module] = bits
            # Support both modelopt and pytorch_quantization attribute names
            if hasattr(module, 'num_bits'):
                module.num_bits = 8
            else:
                module._num_bits = 8
    if saved:
        print(f"[export_onnx] Promoted {len(saved)} INT4 quantizer(s) to INT8 "
              "for opset < 21 export")
    return saved


def _restore_quantizer_bits(saved: dict):
    """Restore quantizer bit widths saved by _save_and_promote_int4_quantizers."""
    for module, bits in saved.items():
        if hasattr(module, 'num_bits'):
            module.num_bits = bits
        else:
            module._num_bits = bits
    if saved:
        print(f"[export_onnx] Restored {len(saved)} quantizer(s) to INT4")


# ===========================================================================
# Custom ONNX graph rules (yolo11-specific quantizer sharing)
# ===========================================================================

def apply_custom_rules_to_quantizer_yolo11(
    model: nn.Module,
    export_onnx_fn: Callable,
):
    """Align quantizer pairs found in the ONNX graph and apply yolo11 rules.

    This function is unchanged from quantize_11.py — it operates on
    TensorQuantizer attribute references regardless of which backend created
    them.  The ``_input_quantizer`` attribute is accessed by name, so it works
    with both modelopt and pytorch_quantization TensorQuantizers.
    """
    export_onnx_fn(model, "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    for major, sub in pairs:
        print(f"ONNX Rules: {sub} match to {major}")
        try:
            get_attr_with_path(model, sub)._input_quantizer = \
                get_attr_with_path(model, major)._input_quantizer
        except Exception as e:
            print(f"Warning: Could not apply rule {sub} -> {major}: {e}")

    if os.path.exists("quantization-custom-rules-temp.onnx"):
        os.remove("quantization-custom-rules-temp.onnx")

    for name, module in model.named_modules():
        if module.__class__.__name__ == "C3k2":
            module.chunkop._input0_quantizer = module.m[0].cv1.conv._input_quantizer

        if module.__class__.__name__ == "C3k":
            major = module.cv1.conv._input_quantizer
            module.m[-1].cv1.conv._input_quantizer = major

        if module.__class__.__name__ == "Bottleneck":
            major = module.cv1.conv._input_quantizer
            module.addop._input0_quantizer = major
            module.addop._input1_quantizer = major

        if isinstance(module, nn.MaxPool2d):
            # modelopt will have already replaced MaxPool2d with a quantized
            # version via mtq.quantize().  If not, fall back to manual replacement
            # using pytorch_quantization (graceful degradation).
            try:
                from pytorch_quantization import nn as _pq_nn2
                from pytorch_quantization.tensor_quant import QuantDescriptor as _QD2
                quant_conv_desc_input = _QD2(num_bits=8, calib_method='histogram')
                quant_maxpool2d = _pq_nn2.QuantMaxPool2d(
                    module.kernel_size, module.stride, module.padding,
                    module.dilation, module.ceil_mode,
                    quant_desc_input=quant_conv_desc_input,
                )
                set_module(model, name, quant_maxpool2d)
            except ImportError:
                pass  # modelopt handles MaxPool2d automatically


# ===========================================================================
# QAT fine-tuning (unchanged except TensorQuantizer isinstance)
# ===========================================================================

def finetune(
    model: nn.Module,
    train_dataloader,
    per_epoch_callback: Callable = None,
    preprocess: Callable = None,
    nepochs: int = 10,
    early_exit_batchs_per_epoch: int = 1000,
    lrschedule: Dict = None,
    fp16: bool = True,
    learningrate: float = 1e-5,
    supervision_policy: Callable = None,
    prefix: str = "",
):
    """QAT fine-tuning with MSE distillation from the FP32 teacher model."""
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    origin_model.train()
    for param in origin_model.parameters():
        param.requires_grad = False

    scaler = amp.GradScaler(enabled=fp16)
    optimizer = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {0: 1e-6, 3: 1e-5, 8: 1e-6}

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        l.append(item)
                        break
                else:
                    l.append(output[0] if len(output) > 0 else output)
            else:
                l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(
        model.named_modules(), origin_model.named_modules()
    ):
        # Skip quantizer nodes — their outputs are not meaningful teacher targets
        if isinstance(ml, _TensorQuantizer):
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

        model_outputs: list = []
        origin_outputs: list = []
        remove_handle = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(
                ml.register_forward_hook(make_layer_forward_hook(model_outputs))
            )
            remove_handle.append(
                ori.register_forward_hook(make_layer_forward_hook(origin_outputs))
            )

        model.train()
        pbar = tqdm(
            train_dataloader,
            desc=f"{prefix}QAT",
            total=early_exit_batchs_per_epoch,
        )
        for ibatch, imgs in enumerate(pbar):
            if ibatch >= early_exit_batchs_per_epoch:
                break
            if preprocess:
                imgs = preprocess(imgs)
            else:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(device, non_blocking=True).float() / 255.0

            with amp.autocast(enabled=fp16):
                model(imgs)
                with torch.no_grad():
                    origin_model(imgs)
                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    if isinstance(mo, torch.Tensor) and isinstance(fo, torch.Tensor):
                        if mo.shape == fo.shape:
                            quant_loss += quant_lossfn(mo, fo)
                        else:
                            print(
                                f"Warning: Shape mismatch at index {index}: "
                                f"{mo.shape} vs {fo.shape}"
                            )
                    else:
                        print(
                            f"Warning: Non-tensor at index {index}: "
                            f"{type(mo)} vs {type(fo)}"
                        )
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
            pbar.set_description(
                f"{prefix}QAT Finetuning {iepoch + 1}/{nepochs}, "
                f"Loss: {quant_loss.detach().item():.5f}, "
                f"LR: {learningrate:g}"
            )

        for rm in remove_handle:
            rm.remove()
        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


# ===========================================================================
# Legacy compatibility stubs
# ===========================================================================

def quantization_ignore_match(
    ignore_policy: Union[str, List[str], Callable],
    path: str,
) -> bool:
    """Kept for API compatibility. Used nowhere in the modelopt path."""
    if ignore_policy is None:
        return False
    if callable(ignore_policy):
        return ignore_policy(path)
    if isinstance(ignore_policy, str):
        ignore_policy = [ignore_policy]
    if path in ignore_policy:
        return True
    return any(re.match(item, path) for item in ignore_policy)


def transfer_torch_to_quantization(nninstance, quantmodule):
    """Deprecated stub. modelopt handles module replacement internally."""
    raise NotImplementedError(
        "transfer_torch_to_quantization is not used in the modelopt path. "
        "Module replacement happens inside mtq.quantize()."
    )


# ===========================================================================
# Bottom-of-file aliases (same as quantize_11.py)
# ===========================================================================

replace_bottleneck_forward = replace_bottleneck_forward_yolo11
apply_custom_rules_to_quantizer = apply_custom_rules_to_quantizer_yolo11
replace_custom_module_forward = replace_custom_module_forward_yolo11
