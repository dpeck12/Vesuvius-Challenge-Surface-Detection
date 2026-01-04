from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # Tiling configuration for 3D volumes (Z, Y, X)
    tile_shape: Tuple[int, int, int] = (64, 128, 128)
    overlap: Tuple[int, int, int] = (16, 32, 32)

    # Inference threshold for binarization of probabilities
    threshold: float = 0.5

    # Output mask dtype (must match Kaggle train mask dtype for submissions)
    # Common options: "uint8", "int16". Adjust as needed.
    output_dtype: str = "uint8"

    # Postprocessing
    min_component_size: int = 500
    morph_iterations: int = 1


DEFAULT_CONFIG = Config()