import numpy as np

def winsorization_min_max(band, values:bool = False) -> tuple:

    # calculatees 1st and 99th
    p1 = np.nanpercentile(band, 1)
    p99 = np.nanpercentile(band, 99)

    if type(values) == bool:
        if values:
            return (p1, p99)

        clipped_band = np.clip(band, p1, p99)

        rescaled_band = (clipped_band - p1) / (p99 - p1)

        return rescaled_band

    else:
        clipped_band = np.clip(band, values[0], values[1])

        rescaled_band = (clipped_band - p1) / (p99 - p1)

        return rescaled_band
