import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (avg + max pool fusion)."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1) * x


class SpatialAttention(nn.Module):
    """Spatial attention using avg + max pooled channel maps."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad      = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attn) * x


class MSSA(nn.Module):
    """
    Multi-Scale Spatial Attention Module.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = in_channels // 4

        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU()
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.act          = nn.SiLU()

    def forward(self, x):
        s1    = self.scale1(x)
        s2    = self.scale2(x)
        s3    = self.scale3(x)
        fused = self.fuse(torch.cat([s1, s2, s3], dim=1))
        out   = self.channel_attn(fused)
        out   = self.spatial_attn(out)
        return self.act(out + x)

def inject_mssa(yolo_model, device='cpu'):
    """
    Injects MSSA after each C2f block in the YOLOv8 neck (layers 11+).
    """
    from ultralytics.nn.modules import C2f
    
    seq = yolo_model.model.model
    injected = []

    for idx, layer in enumerate(seq):
        if isinstance(layer, C2f) and idx >= 11:
            try:
                in_ch = layer.cv2.conv.out_channels
            except AttributeError:
                continue
            mssa = MSSA(in_channels=in_ch).to(device)
            new_seq = nn.Sequential(layer, mssa)
            # Crucial: Copy Ultralytics metadata to the new Sequential block
            new_seq.i = idx
            new_seq.f = layer.f
            new_seq.type = layer.type
            
            seq[idx] = new_seq
            injected.append(idx)
    
    if injected:
        print(f"MSSA injected at neck layers: {injected}")
    else:
        print("No layers injected with MSSA. Check model architecture.")
    
    return yolo_model
