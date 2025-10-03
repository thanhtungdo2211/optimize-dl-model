from thop import profile
from ultralytics.nn.modules import C2f, Conv, Bottleneck  
from pathlib import Path
import torch.nn as nn
from ultralytics import YOLO
import torch

yolo_model = YOLO('yolov8s.pt').model

flops, params = profile(yolo_model, inputs=(torch.randn(1, 3, 640, 640),))
print("Params base:", params, "FLOPs:", flops)




def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            try:
                setattr(c2f_v2, attr_name, attr_value)
            except Exception:
                pass

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            try:
                transfer_weights(child_module, c2f_v2)
            except Exception:
                pass
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)
            
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
ckpt = torch.load('runs/detect/step_6_finetune/weights/best.pt', map_location=device)
pruning_model = ckpt['model'] if 'model' in ckpt else ckpt
pruning_model = pruning_model.to(device).float().eval()

flops, params = profile(pruning_model, inputs=(torch.randn(1, 3, 640, 640).to(device),))
print("Params after pruning:", params, "FLOPs:", flops)