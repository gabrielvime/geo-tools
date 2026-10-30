import numpy as np

def winsorization_min_max(band):

    # calculatees 1st and 99th
    p1 = np.nanpercentile(band, 1)
    p99 = np.nanpercentile(band, 99)

    clipped_band = np.clip(band, p1, p99)

    rescaled_band = (clipped_band - p1) / (p99 - p1)

    return rescaled_band
