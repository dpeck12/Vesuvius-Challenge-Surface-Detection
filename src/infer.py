import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import DEFAULT_CONFIG
from src.data import read_volume, write_mask, list_volumes
from src.model import UNet3D
from src.postprocess import binarize, morphological_cleanup, remove_small_components
from src.tiling import VolumeTiler


def predict_volume(vol: np.ndarray, model: UNet3D, device: torch.device, cfg=DEFAULT_CONFIG) -> np.ndarray:
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


def run_inference(images_dir: str, output_dir: str, checkpoint: str | None):
    cfg = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = list_volumes(images_dir)
    if not image_paths:
        raise RuntimeError(f"No .tif images found in {images_dir}")

    model: UNet3D | None
    if checkpoint:
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
            # Heuristic fallback: use normalized intensities as probabilities
            probs = vol_norm.astype(np.float32)
        else:
            probs = predict_volume(vol_norm, model, device, cfg)
        mask = binarize(probs, cfg.threshold)
        mask = morphological_cleanup(mask, iterations=cfg.morph_iterations)
        mask = remove_small_components(mask, min_size=cfg.min_component_size)
        # Write with configured dtype (adjust to train mask dtype when known)
        out_path = out_dir / f"{img_path.stem}.tif"
        write_mask(str(out_path), mask, dtype=cfg.output_dtype)

    print(f"Wrote masks to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()
    run_inference(args.images_dir, args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()