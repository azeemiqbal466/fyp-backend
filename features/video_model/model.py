import os

import torch
import torch.nn as nn
import timm

# ─────────────────────────── DEVICE ─────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("GPU not detected. Using CPU.")

print("Using:", device)


# ─────────────────────────── MODEL DEFINITION ───────────────────
class VideoManipulationModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )

        old_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)

        self.backbone.conv_stem = new_conv

        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, -1)
        features = features.mean(dim=1)
        return self.classifier(features)


# ─────────────────────────── SINGLETON LOADER ───────────────────
_model_instance: VideoManipulationModel | None = None


def video_manipulation_model() -> VideoManipulationModel:
    """
    Returns the singleton VideoManipulationModel, loading weights on the
    first call.  Subsequent calls return the already-loaded instance so
    the checkpoint is never read from disk more than once.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model = VideoManipulationModel().to(device)

    print("Parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # ── Load checkpoint ────────────────────────────────────────
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_visual_manipulation.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}\n"
            "Place 'best_visual_manipulation.pth' next to model.py."
        )

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Video Model loaded successfully.")
    _model_instance = model
    return _model_instance


# ─────────────────────────── EAGER WARM-UP ──────────────────────
# Load the model when this module is first imported so it is ready
# before the first HTTP request arrives.
try:
    video_manipulation_model()
except FileNotFoundError as _e:
    # Warn but do not crash — the router will surface the error
    # on the first /predict call.
    import warnings
    warnings.warn(str(_e), stacklevel=1)