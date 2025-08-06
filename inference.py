import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from model import DeepLabV3Wrapper
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple
from skimage.measure import label, regionprops

# Configuration
MODEL_PATH     = "/checkpoint/trained_deeplabv3checkpoint.pth"
SAM_CHECKPOINT = "sam_checkpoint/sam_vit_h_4b8939.pth"
QUERY_DIR      = "/query"
OUTPUT_DIR     = "/output"
SIZE           = (512, 512)
NUM_POINTS     = 20
EXTENT_X       = 275   
EXTENT_Y       = 275
DEVICE         = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sample_spread_points(coords: np.ndarray, n_points: int) -> np.ndarray:
    """
    Pick n_points from coords as far apart as possible, using precomputed distances.
    """
    M = coords.shape[0]
    if M == 0 or n_points == 0:
        return np.zeros((0, 2), dtype=int)
    # Compute pairwise squared distances (M x M)
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    # Greedy farthest-first traversal
    chosen = [np.random.randint(0, M)]
    min_dist2 = dist2[chosen[0]].copy()
    for _ in range(1, min(n_points, M)):
        idx = np.argmax(min_dist2)
        chosen.append(idx)
        min_dist2 = np.minimum(min_dist2, dist2[idx])
    return coords[chosen]


def sample_top_bottom_points(
    prob_map: np.ndarray,
    num_points: int,
    threshold: float = 0.9,
    pool_factor: int = 10
):
    """
    Sample num_points FG and BG points spread out across high (>threshold)
    and low (<1-threshold) confidence regions. Randomly subsample a pool
    of size num_points*pool_factor before spreading.
    """
    h, w = prob_map.shape
    flat = prob_map.flatten()

    # Identify candidate indices
    fg_idxs = np.where(flat >= threshold)[0]
    bg_idxs = np.where(flat <= (1 - threshold))[0]

    M_fg = len(fg_idxs)
    M_bg = len(bg_idxs)

    # Subsample pools randomly
    pool_size = num_points * pool_factor
    fg_pool = fg_idxs if M_fg <= pool_size else np.random.choice(fg_idxs, pool_size, replace=False)
    bg_pool = bg_idxs if M_bg <= pool_size else np.random.choice(bg_idxs, pool_size, replace=False)

    # Convert flat indices to (x, y) coords
    coords_all = lambda idxs: np.stack((idxs % w, idxs // w), axis=1)
    fg_coords = coords_all(fg_pool)
    bg_coords = coords_all(bg_pool)

    # Spread out samples using farthest-first
    fg_spread = sample_spread_points(fg_coords, num_points)
    bg_spread = sample_spread_points(bg_coords, num_points)

    coords = np.vstack([fg_spread, bg_spread])
    labels = np.array([1] * len(fg_spread) + [0] * len(bg_spread))
    return coords, labels

def clean_pred_map(prob_map: np.ndarray, min_area: int = 300) -> np.ndarray:
    binary = (prob_map > 0.5).astype(np.uint8)
    lbl = label(binary)
    out = np.zeros_like(binary)
    for region in regionprops(lbl):
        if region.area >= min_area:
            out[lbl == region.label] = 1
    return prob_map * out

def filter_components_by_extent(
    prob_map: np.ndarray,
    min_xdist: int,
    min_ydist: int
) -> np.ndarray:
    """
    Keep only those connected components whose bounding box width >= min_xdist
    AND height >= min_ydist. Returns the original prob_map masked to those regions.
    """
    # binarize at 0.5
    binary = (prob_map > 0.8).astype(np.uint8)
    lbl = label(binary)
    mask = np.zeros_like(binary)
    for region in regionprops(lbl):
        minr, minc, maxr, maxc = region.bbox
        width  = maxc - minc
        height = maxr - minr
        if width >= min_xdist or height >= min_ydist:
            mask[lbl == region.label] = 1
    # zero‐out small components in the prob_map
    return prob_map * mask



def main():

    # Load models
    dl_model = DeepLabV3Wrapper(in_channels=1, out_channels=1)
    dl_model = dl_model.to(DEVICE) 
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    dl_model.load_state_dict(state)
    dl_model.eval()

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE).eval()
    predictor = SamPredictor(sam)

    # Warm-up
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    predictor.set_image(dummy)
    predictor.predict(
        point_coords=np.array([[0, 0]]),
        point_labels=np.array([0]),
        multimask_output=False
    )

    tf = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # scale to [-1, 1]
    ])

    all_files = sorted(
        f for f in os.listdir(QUERY_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    )
    for fname in tqdm(all_files, desc="Inference", unit="img"):

        # 1) Load image (BGR → RGB)
        img_path = os.path.join(QUERY_DIR, fname)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # 2) Preprocess & DeepLab inference → small probability map
        img_t = tf(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = dl_model(img_t)[0, 0]
            pred_resized = torch.sigmoid(logits).cpu().numpy()

        # 3) Upsample DeepLab output to original size
        pred_map = cv2.resize(pred_resized, (W, H), interpolation=cv2.INTER_LINEAR)

        # 4) Clean up the mask: remove tiny CCs and filter by extent
        pred_map = clean_pred_map(pred_map, min_area=1000)
        pred_map = filter_components_by_extent(pred_map, EXTENT_X, EXTENT_Y)

        # 5) Sample far-apart foreground/background points
        pts, labs = sample_top_bottom_points(pred_map, NUM_POINTS)

        # 6) Refine with SAM (mask resolution matches img_rgb, no resize needed)
        predictor.set_image(img_rgb)
        masks, _, _ = predictor.predict(
            point_coords=pts,
            point_labels=labs,
            multimask_output=False
        )

        # 7) Convert to uint8 & save
        mask  = masks[0].astype(np.uint8) * 255
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_sam.png")
        cv2.imwrite(out_path, mask)


if __name__ == "__main__":
    main()
