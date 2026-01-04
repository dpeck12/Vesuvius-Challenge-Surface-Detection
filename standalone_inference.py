# Standalone Kaggle-ready inference + submission packing script
# No repo imports; minimal package bootstrap; optional torch model fallback

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# --- Optional lightweight package bootstrap ---
def ensure_packages(packages: List[str]) -> None:
    missing = []
    for p in packages:
        try:
            __import__(p)
        except Exception:
            missing.append(p)
    if missing:
        try:
            print("Installing:", missing)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U"] + missing)
        except Exception as e:
            print("Package install warning:", e)

# Core dependencies; torch is optional
ensure_packages(["numpy", "tifffile", "scipy", "tqdm"])

import numpy as np
import zipfile
import tifffile as tiff
from tqdm import tqdm
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, label

# Torch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# --- Config ---
class Config:
    # Tiling configuration for 3D volumes (Z, Y, X)
    tile_shape: Tuple[int, int, int] = (64, 128, 128)
    overlap: Tuple[int, int, int] = (16, 32, 32)
    # Inference threshold for binarization of probabilities
    threshold: float = 0.5
    # Output mask dtype
    output_dtype: str = "uint8"
    # Postprocessing
    min_component_size: int = 500
    morph_iterations: int = 1


DEFAULT_CONFIG = Config()


# --- Data IO ---
def read_volume(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def write_mask(path: str, mask: np.ndarray, dtype: str = "uint8") -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_np = mask.astype(getattr(np, dtype))
    tiff.imwrite(str(out_path), mask_np)


def list_volumes(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    return sorted(p.glob("*.tif"))


# --- Postprocess ---
def binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.uint8)


def morphological_cleanup(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    struct = generate_binary_structure(rank=3, connectivity=2)
    m = mask.astype(bool)
    if iterations > 0:
        m = binary_opening(m, structure=struct, iterations=iterations)
        m = binary_closing(m, structure=struct, iterations=iterations)
    return m.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    struct = generate_binary_structure(rank=3, connectivity=2)
    labeled, ncomp = label(mask.astype(bool), structure=struct)
    if ncomp == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[0] = False
    keep[1:] = sizes[1:] >= min_size
    out = keep[labeled]
    return out.astype(np.uint8)


# --- Tiling ---
def hann_window_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = shape
    wz = np.hanning(max(z, 2))
    wy = np.hanning(max(y, 2))
    wx = np.hanning(max(x, 2))
    wz = wz / (wz.max() if wz.max() > 0 else 1)
    wy = wy / (wy.max() if wy.max() > 0 else 1)
    wx = wx / (wx.max() if wx.max() > 0 else 1)
    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return w.astype(np.float32)


def compute_slices(start: int, size: int) -> slice:
    return slice(start, start + size)


class VolumeTiler:
    def __init__(self, tile_shape: Tuple[int, int, int], overlap: Tuple[int, int, int]):
        self.tile_shape = tile_shape
        self.overlap = overlap
        self.window = hann_window_3d(tile_shape)

    def _grid(self, vol_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        tz, ty, tx = self.tile_shape
        oz, oy, ox = self.overlap
        sz, sy, sx = vol_shape
        step_z = max(tz - oz, 1)
        step_y = max(ty - oy, 1)
        step_x = max(tx - ox, 1)
        starts = []
        for z in range(0, max(sz - tz, 0) + 1, step_z):
            for y in range(0, max(sy - ty, 0) + 1, step_y):
                for x in range(0, max(sx - tx, 0) + 1, step_x):
                    starts.append((z, y, x))
        if starts:
            last_z, last_y, last_x = starts[-1]
            if last_z + tz < sz:
                for y in range(0, max(sy - ty, 0) + 1, step_y):
                    for x in range(0, max(sx - tx, 0) + 1, step_x):
                        starts.append((sz - tz, y, x))
            if last_y + ty < sy:
                for z in range(0, max(sz - tz, 0) + 1, step_z):
                    for x in range(0, max(sx - tx, 0) + 1, step_x):
                        starts.append((z, sy - ty, x))
            if last_x + tx < sx:
                for z in range(0, max(sz - tz, 0) + 1, step_z):
                    for y in range(0, max(sy - ty, 0) + 1, step_y):
                        starts.append((z, y, sx - tx))
        return starts

    def tile(self, volume: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
        tz, ty, tx = self.tile_shape
        tiles = []
        for (z0, y0, x0) in self._grid(volume.shape):
            zsl = compute_slices(z0, tz)
            ysl = compute_slices(y0, ty)
            xsl = compute_slices(x0, tx)
            tiles.append((volume[zsl, ysl, xsl], (z0, y0, x0)))
        return tiles

    def untile(self, tile_preds: List[Tuple[np.ndarray, Tuple[int, int, int]]], full_shape: Tuple[int, int, int]) -> np.ndarray:
        tz, ty, tx = self.tile_shape
        out = np.zeros(full_shape, dtype=np.float32)
        wsum = np.zeros(full_shape, dtype=np.float32)
        w = self.window
        for tile_pred, (z0, y0, x0) in tile_preds:
            zsl = compute_slices(z0, tz)
            ysl = compute_slices(y0, ty)
            xsl = compute_slices(x0, tx)
            out[zsl, ysl, xsl] += tile_pred.astype(np.float32) * w
            wsum[zsl, ysl, xsl] += w
        mask = wsum > 0
        out[mask] /= wsum[mask]
        return out


# --- Model (optional) ---
if TORCH_AVAILABLE:
    def conv3x3(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    class Down(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = conv3x3(in_ch, out_ch)
            self.pool = nn.MaxPool3d(2)
        def forward(self, x: torch.Tensor):
            x = self.conv(x)
            skip = x
            x = self.pool(x)
            return x, skip

    class Up(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = conv3x3(in_ch, out_ch)
        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(x, [0, diffX, 0, diffY, 0, diffZ])
            x = torch.cat([skip, x], dim=1)
            x = self.conv(x)
            return x

    class UNet3D(nn.Module):
        def __init__(self, in_channels: int = 1, base: int = 16):
            super().__init__()
            self.down1 = Down(in_channels, base)
            self.down2 = Down(base, base * 2)
            self.bottom = conv3x3(base * 2, base * 4)
            self.up2 = Up(base * 4, base * 2)
            self.up1 = Up(base * 2, base)
            self.out = nn.Conv3d(base, 1, kernel_size=1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x, s1 = self.down1(x)
            x, s2 = self.down2(x)
            x = self.bottom(x)
            x = self.up2(x, s2)
            x = self.up1(x, s1)
            logits = self.out(x)
            return logits


# --- Inference ---
def predict_volume(vol: np.ndarray, model: "UNet3D", device: "torch.device", cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    tiler = VolumeTiler(cfg.tile_shape, cfg.overlap)
    tiles = tiler.tile(vol)
    preds = []
    model.eval()
    with torch.no_grad():
        for tile, start in tiles:
            t = torch.from_numpy(tile[None, None, ...]).float().to(device)
            logits = model(t)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            preds.append((probs, start))
    full_probs = tiler.untile(preds, vol.shape)
    return full_probs


def run_inference(images_dir: str, output_dir: str, checkpoint: Optional[str] = None) -> None:
    cfg = DEFAULT_CONFIG
    image_paths = list_volumes(images_dir)
    if not image_paths:
        raise RuntimeError(f"No .tif images found in {images_dir}")

    use_model = TORCH_AVAILABLE and (checkpoint is not None) and os.path.isfile(checkpoint)
    if use_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet3D(in_channels=1, base=16).to(device)
        sd = torch.load(checkpoint, map_location=device)
        model.load_state_dict(sd)
    else:
        model = None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Infer"):
        vol = read_volume(str(img_path))
        vol_norm = (vol - vol.min()) / max((vol.max() - vol.min()), 1e-6)
        if model is None:
            probs = vol_norm.astype(np.float32)  # heuristic fallback
        else:
            probs = predict_volume(vol_norm, model, device, cfg)
        mask = binarize(probs, cfg.threshold)
        mask = morphological_cleanup(mask, iterations=cfg.morph_iterations)
        mask = remove_small_components(mask, min_size=cfg.min_component_size)
        out_path = out_dir / f"{img_path.stem}.tif"
        write_mask(str(out_path), mask, dtype=cfg.output_dtype)

    print(f"Wrote masks to {out_dir}")


# --- Pack submission ---
def pack_submission(masks_dir: str, output_zip: str) -> None:
    src = Path(masks_dir)
    out = Path(output_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(src.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif masks found in {src}")
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in tif_files:
            zf.write(f, arcname=f.name)
    print(f"Created {out} with {len(tif_files)} files")


# --- CLI ---
import argparse

def auto_detect_images_dir() -> Optional[str]:
    candidate_dirs = [
        "/kaggle/input/vesuvius-challenge-surface-detection/test_images",
        "/kaggle/input/vesuvius-challenge-surface-detection/test",
        "./data/test_images",
        "./data/raw/test_images",
        "./data/raw/test",
    ]
    for d in candidate_dirs:
        if os.path.isdir(d):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description="Standalone Vesuvius inference + submission packer")
    ap.add_argument("--images_dir", type=str, default=None, help="Directory containing test .tif volumes")
    ap.add_argument("--output_masks_dir", type=str, default="./submission_masks", help="Directory to write mask .tif files")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to torch checkpoint (optional)")
    ap.add_argument("--submission_zip", type=str, default="./submission.zip", help="Output zip path for Kaggle submission")
    # Use parse_known_args to ignore Jupyter/Colab kernel args like -f <connection_file>
    args, _ = ap.parse_known_args()

    images_dir = args.images_dir or auto_detect_images_dir()
    if images_dir is None:
        raise RuntimeError("Test images directory not found. Provide --images_dir explicitly.")

    print("IMAGES_DIR:", images_dir)
    print("CHECKPOINT:", args.checkpoint, "TORCH_AVAILABLE:", TORCH_AVAILABLE)

    run_inference(images_dir, args.output_masks_dir, args.checkpoint)
    pack_submission(args.output_masks_dir, args.submission_zip)

    size_mb = round(os.path.getsize(args.submission_zip) / 1e6, 3) if os.path.exists(args.submission_zip) else 0
    print("Submission size (MB):", size_mb)

    # Sanity output
    masks = sorted(Path(args.output_masks_dir).glob("*.tif"))
    if masks:
        m = tiff.imread(str(masks[0]))
        print("Mask shape:", m.shape, "dtype:", m.dtype, "min/max:", int(m.min()), int(m.max()))
    else:
        print("No masks written in", args.output_masks_dir)


if __name__ == "__main__":
    main()
