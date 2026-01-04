from typing import List, Tuple

import numpy as np


def hann_window_3d(shape: Tuple[int, int, int]) -> np.ndarray:
    """Create a separable 3D Hann window for overlap-weighted stitching."""
    z, y, x = shape
    wz = np.hanning(max(z, 2))
    wy = np.hanning(max(y, 2))
    wx = np.hanning(max(x, 2))
    # Normalize each to max=1 to avoid tiny scales
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

        # Ensure we cover the tail ends exactly
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
        """Return list of (tile_array, (z,y,x_start)) for a 3D (Z,Y,X) volume."""
        tz, ty, tx = self.tile_shape
        tiles = []
        for (z0, y0, x0) in self._grid(volume.shape):
            zsl = compute_slices(z0, tz)
            ysl = compute_slices(y0, ty)
            xsl = compute_slices(x0, tx)
            tiles.append((volume[zsl, ysl, xsl], (z0, y0, x0)))
        return tiles

    def untile(self, tile_preds: List[Tuple[np.ndarray, Tuple[int, int, int]]], full_shape: Tuple[int, int, int]) -> np.ndarray:
        """Stitch overlapping tile predictions via weighted averaging with Hann window."""
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
        # Avoid division by zero
        mask = wsum > 0
        out[mask] /= wsum[mask]
        return out