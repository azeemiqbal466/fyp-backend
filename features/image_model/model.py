import timm
import torch
import os

# -----------------------------------
# DEVICE
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# CONFIG
# -----------------------------------
IMG_SIZE = 224


Base_Dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(Base_Dir, "best_convnext_fractal.pth")

# -----------------------------------
# BUILD MODEL
# -----------------------------------
def build_model():
    model = timm.create_model(
        "convnext_small",
        pretrained=False,
        num_classes=2,
        in_chans=4,
        depths=(3, 3, 9, 3),
    )
    return model


# -----------------------------------
# LOAD MODEL (singleton)
# -----------------------------------
def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model = model.to(device)
    model.eval()
    print("Image Model loaded successfully.")
    return model


# Module-level singleton — imported once, reused across requests
model = load_model()