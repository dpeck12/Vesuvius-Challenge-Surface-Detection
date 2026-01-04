from pathlib import Path
from typing import List

import numpy as np
import tifffile as tiff


def read_volume(path: str) -> np.ndarray:
    """Read a 3D volume from a .tif file as (Z, Y, X) numpy array."""
    arr = tiff.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # Ensure dtype stays as read for inference heuristics; typically integer CT intensities.
    return arr


def write_mask(path: str, mask: np.ndarray, dtype: str = "uint8") -> None:
    """Write a 3D mask volume to .tif with given dtype (e.g., 'uint8')."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_np = mask.astype(getattr(np, dtype))
    tiff.imwrite(str(out_path), mask_np)


def list_volumes(dir_path: str) -> List[Path]:
    """List .tif files in a directory (non-recursive)."""
    p = Path(dir_path)
    return sorted(p.glob("*.tif"))