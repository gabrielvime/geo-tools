import numpy as np


def winsorization_min_max(
    band: np.ndarray,
    values: tuple[float | np.ndarray, float | np.ndarray] | bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Applies channel-wise 1st-99th percentile winsorization and min-max scaling.

    Args:
        band: Input 2D (H, W) or 3D HWC (H, W, C) array.
        values: If True, returns the (p1, p99) bounds per channel without scaling.
            If False, computes bounds from `band` and scales to [0, 1].
            If a tuple of (low, high) bounds is passed, clips and scales using them.

    Returns:
        Tuple of (p1, p99) if `values` is True, otherwise the rescaled float32 array.
    """
    spatial_axes = (0, 1) if band.ndim >= 2 else None

    if isinstance(values, bool):
        p1, p99 = np.nanpercentile(band, [1, 99], axis=spatial_axes)
        if values:
            return (p1, p99)
        low, high = p1, p99
    else:
        low, high = values

    clipped_band = np.clip(band, low, high)
    denom = np.where((high - low) == 0, 1.0, high - low)
    rescaled_band = (clipped_band - low) / denom

    return rescaled_band.astype(np.float32)