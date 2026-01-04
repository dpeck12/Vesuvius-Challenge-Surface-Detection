import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import DEFAULT_CONFIG
from src.data import read_volume
from src.model import UNet3D
from src.tiling import VolumeTiler


class TileDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.cfg = DEFAULT_CONFIG
        self.tiler = VolumeTiler(self.cfg.tile_shape, self.cfg.overlap)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        vol = read_volume(str(self.image_paths[idx]))
        # For demo: create pseudo-labels as gradient threshold (placeholder)
        vol_norm = (vol - vol.min()) / max((vol.max() - vol.min()), 1e-6)
        label = (vol_norm > 0.5).astype(np.uint8)
        # Take a random tile for training
        tiles = self.tiler.tile(vol_norm)
        if not tiles:
            tiles = [(vol_norm, (0, 0, 0))]
        tile, _ = tiles[np.random.randint(0, len(tiles))]
        z, y, x = tile.shape
        # Prepare tensors
        img_t = torch.from_numpy(tile[None, None, ...]).float()
        lbl_t = torch.from_numpy((tile > 0.5)[None, None, ...].astype(np.float32))
        return img_t, lbl_t


def train(args):
    img_dir = Path(args.images_dir)
    image_paths = sorted(img_dir.glob("*.tif"))
    if not image_paths:
        raise RuntimeError(f"No .tif images found in {img_dir}")

    ds = TileDataset(image_paths)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, base=16).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img_t, lbl_t in pbar:
            img_t = img_t.to(device)
            lbl_t = lbl_t.to(device)
            logits = model(img_t)
            loss = loss_fn(logits, lbl_t)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))

    ckpt_path = Path(args.output_dir) / "model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()