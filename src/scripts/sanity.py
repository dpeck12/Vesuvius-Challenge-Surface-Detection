from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.config import DEFAULT_CONFIG
from src.data import write_mask
from src.tiling import VolumeTiler
from src.postprocess import binarize, morphological_cleanup, remove_small_components


def generate_synthetic_volume(shape=(64, 128, 128)):
    z, y, x = np.indices(shape)
    # Synthetic thin surface: a wavy plane around z=shape[0]//2
    z0 = shape[0] // 2 + (10 * np.sin(2 * np.pi * y / 64) * np.sin(2 * np.pi * x / 64)).astype(int)
    surface = (np.abs(z - z0) <= 1).astype(np.float32)
    # Add background noise
    noise = 0.1 * np.random.randn(*shape).astype(np.float32)
    vol = surface + noise
    # Normalize to [0,1]
    vol = (vol - vol.min()) / max((vol.max() - vol.min()), 1e-6)
    return vol


def main():
    cfg = DEFAULT_CONFIG
    vol = generate_synthetic_volume(cfg.tile_shape)
    tiler = VolumeTiler(cfg.tile_shape, cfg.overlap)
    tiles = tiler.tile(vol)
    preds = []
    for tile, start in tqdm(tiles, desc="Synthetic predict"):
        # Simple heuristic: local threshold
        probs = tile
        preds.append((probs, start))
    full_probs = tiler.untile(preds, vol.shape)
    mask = binarize(full_probs, cfg.threshold)
    mask = morphological_cleanup(mask, iterations=cfg.morph_iterations)
    mask = remove_small_components(mask, min_size=cfg.min_component_size)
    out_dir = Path("./synthetic_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_mask.tif"
    write_mask(str(out_path), mask, dtype=cfg.output_dtype)
    print(f"Synthetic mask written to {out_path}")


if __name__ == "__main__":
    main()