from typing import Tuple

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, label


def binarize(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.uint8)


def morphological_cleanup(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply light opening followed by closing with a 3D connectivity structure."""
    struct = generate_binary_structure(rank=3, connectivity=2)  # 18-connectivity
    m = mask.astype(bool)
    if iterations > 0:
        m = binary_opening(m, structure=struct, iterations=iterations)
        m = binary_closing(m, structure=struct, iterations=iterations)
    return m.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
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