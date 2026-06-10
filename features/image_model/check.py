import os
import cv2
import timm
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------------
# DEVICE
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# CONFIG
# -----------------------------------
IMG_SIZE = 224


MODEL_PATH = "best_convnext_fractal.pth"

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
# LOAD MODEL
# -----------------------------------
model = build_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")

# -----------------------------------
# TRANSFORMS
# -----------------------------------
train_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.10,
        rotate_limit=10,
        p=0.5
    ),
    A.GaussianBlur(p=0.15),
    A.ImageCompression(
        quality_range=(40, 100),
        p=0.3
    ),
    A.Normalize(),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# -----------------------------------
# FRACTAL MAP
# -----------------------------------
def fractal_map(gray, patch=16):

    h, w = gray.shape

    H = (h // patch) * patch
    W = (w // patch) * patch

    gray = gray[:H, :W]

    out = np.zeros((H, W), dtype=np.float32)

    thresholds = [64, 96, 128, 160, 192]

    for y in range(0, H, patch):
        for x in range(0, W, patch):

            p = gray[y:y+patch, x:x+patch]

            counts = []

            for t in thresholds:

                bw = (p > t).astype(np.uint8)

                counts.append(bw.sum())

            score = np.std(counts) / (patch * patch + 1e-6)

            out[y:y+patch, x:x+patch] = score

    out = out - out.min()

    out = out / (out.max() + 1e-6)

    return out

# -----------------------------------
# REQUIRED ALREADY:
# model, device, IMG_SIZE, val_tfms
# -----------------------------------
model.eval()

# -----------------------------------
# ENSURE OUTPUT DIR
# -----------------------------------
def _ensure_dir(d):

    if d and (not os.path.exists(d)):

        os.makedirs(d, exist_ok=True)

# -----------------------------------
# BUILD 4-CHANNEL INPUT
# -----------------------------------
def _build_4ch_input(rgb, patch=16):

    """
    RGB (H,W,3) -> (H,W,4)
    """

    H, W = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    f_map = fractal_map(gray, patch=patch)

    f_map = np.nan_to_num(
        f_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # normalize
    fmin, fmax = float(f_map.min()), float(f_map.max())

    if fmax > fmin:

        f_map_norm = (f_map - fmin) / (fmax - fmin)

    else:

        f_map_norm = np.zeros_like(
            f_map,
            dtype=np.float32
        )

    # resize to exact image size
    if f_map_norm.shape[0] != H or f_map_norm.shape[1] != W:

        f_map_norm = cv2.resize(
            f_map_norm,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

    # convert to uint8
    f_u8 = (f_map_norm * 255).astype(np.uint8)

    # add channel dim
    f_u8 = f_u8[:, :, None]

    # concat RGB + fractal
    img4 = np.concatenate([rgb, f_u8], axis=2)

    return img4

# -----------------------------------
# PREDICT
# -----------------------------------
@torch.no_grad()
def _predict_prob_fake(img4):

    """
    img4 -> (H,W,4)
    returns:
        prob_real, prob_fake
    """

    aug = val_tfms(image=img4)

    x = aug["image"].unsqueeze(0).to(device)

    logits = model(x)

    probs = torch.softmax(logits, dim=1)[0]

    probs = probs.detach().cpu().numpy()

    return float(probs[0]), float(probs[1])

# -----------------------------------
# MAIN DETECTOR
# -----------------------------------
def detect_deepfake_best_localization(
    img_path: str,
    patch_for_fractal: int = 16,
    grid: int = 8,
    stride: int | None = None,
    thr: float = 0.60,
    fake_class_index: int = 1,
    save_dir: str = "outputs",
    show_plots: bool = True,
):

    _ensure_dir(save_dir)

    # read image
    bgr = cv2.imread(img_path)

    if bgr is None:

        raise FileNotFoundError(
            f"Could not read image at path: {img_path}"
        )

    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb_resized = cv2.resize(
        rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------------
    # GLOBAL PREDICTION
    # -----------------------------------
    img4_full = _build_4ch_input(
        rgb,
        patch=patch_for_fractal
    )

    prob_real, prob_fake = _predict_prob_fake(img4_full)

    pred = int(prob_fake >= prob_real)

    label = "Fake" if pred == fake_class_index else "Real"

    # -----------------------------------
    # REAL IMAGE
    # -----------------------------------
    if label == "Real":

        hot = np.zeros(
            (IMG_SIZE, IMG_SIZE),
            dtype=np.uint8
        )

        overlay = rgb_resized.copy()

        percent = 0.0

        base = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        overlay_path = os.path.join(
            save_dir,
            f"{base}_prob_overlay.png"
        )

        mask_path = os.path.join(
            save_dir,
            f"{base}_hot_mask.png"
        )

        cv2.imwrite(
            overlay_path,
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        cv2.imwrite(mask_path, hot)

        if show_plots:

            plt.figure(figsize=(14,4))

            plt.subplot(1,3,1)
            plt.title("Input")
            plt.imshow(rgb_resized)
            plt.axis("off")

            plt.subplot(1,3,2)
            plt.title("Result: REAL\nNo suspicious area")
            plt.imshow(rgb_resized)
            plt.axis("off")

            plt.subplot(1,3,3)
            plt.title("Hot Mask\n0%")
            plt.imshow(hot, cmap="gray")
            plt.axis("off")

            plt.show()

            plt.figure(figsize=(6,3))

            plt.title("Probability")

            plt.bar(
                ["Real", "Fake"],
                [prob_real, prob_fake]
            )

            plt.ylim(0,1)

            plt.show()

        return {
            "prediction": "Real",
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "suspicious_area_percent": 0.0,
            "overlay_path": overlay_path,
            "mask_path": mask_path
        }

    # -----------------------------------
    # FAKE IMAGE LOCALIZATION
    # -----------------------------------
    H, W = IMG_SIZE, IMG_SIZE

    patch_size = H // grid

    if stride is None:
        stride = patch_size

    prob_map = np.zeros((H, W), dtype=np.float32)

    count_map = np.zeros((H, W), dtype=np.float32)

    rgb_blur = cv2.GaussianBlur(
        rgb_resized,
        (0,0),
        sigmaX=7,
        sigmaY=7
    )

    for y in range(0, H - patch_size + 1, stride):

        for x in range(0, W - patch_size + 1, stride):

            masked = rgb_blur.copy()

            masked[
                y:y+patch_size,
                x:x+patch_size
            ] = rgb_resized[
                y:y+patch_size,
                x:x+patch_size
            ]

            img4_patch = _build_4ch_input(
                masked,
                patch=patch_for_fractal
            )

            pr, pf = _predict_prob_fake(img4_patch)

            prob_map[
                y:y+patch_size,
                x:x+patch_size
            ] += pf

            count_map[
                y:y+patch_size,
                x:x+patch_size
            ] += 1.0

    # average
    prob_map = prob_map / np.maximum(
        count_map,
        1e-6
    )

    # normalize heatmap
    pmin, pmax = float(prob_map.min()), float(prob_map.max())

    if pmax > pmin:

        heat = (prob_map - pmin) / (pmax - pmin)

    else:

        heat = np.zeros_like(
            prob_map,
            dtype=np.float32
        )

    # suspicious regions
    hot = (prob_map >= thr).astype(np.uint8)

    percent = float(hot.mean() * 100.0)

    # overlay
    rgb_float = rgb_resized.astype(np.float32) / 255.0

    overlay = show_cam_on_image(
        rgb_float,
        heat,
        use_rgb=True
    )

    # save
    base = os.path.splitext(
        os.path.basename(img_path)
    )[0]

    overlay_path = os.path.join(
        save_dir,
        f"{base}_prob_overlay.png"
    )

    mask_path = os.path.join(
        save_dir,
        f"{base}_hot_mask.png"
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        mask_path,
        (hot * 255).astype(np.uint8)
    )

    # plots
    if show_plots:

        plt.figure(figsize=(16,4))

        plt.subplot(1,4,1)
        plt.title("Input")
        plt.imshow(rgb_resized)
        plt.axis("off")

        plt.subplot(1,4,2)
        plt.title("Patch-wise Fake Prob Heatmap")
        plt.imshow(overlay)
        plt.axis("off")

        plt.subplot(1,4,3)
        plt.title(
            f"Hot Mask\nSuspicious Area: {percent:.2f}%"
        )
        plt.imshow(hot, cmap="gray")
        plt.axis("off")

        plt.subplot(1,4,4)
        plt.title("Prob Map")
        plt.imshow(prob_map, cmap="inferno")
        plt.axis("off")

        plt.show()

        plt.figure(figsize=(6,3))

        plt.title("Global Probability")

        plt.bar(
            ["Real", "Fake"],
            [prob_real, prob_fake]
        )

        plt.ylim(0,1)

        plt.show()

    print("Prediction:", label)
    print("Prob Real:", prob_real)
    print("Prob Fake:", prob_fake)
    print("Suspicious Area %:", round(percent, 2))

    print("Saved:")
    print(overlay_path)
    print(mask_path)

    return {
        "prediction": label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "suspicious_area_percent": round(percent, 2),
        "overlay_path": overlay_path,
        "mask_path": mask_path
    }

# -----------------------------------
# USAGE
# -----------------------------------
result = detect_deepfake_best_localization(
    img_path="event video.jpeg",
    grid=8,
    thr=0.60,
    save_dir="outputs",
    show_plots=True
)

print(result)