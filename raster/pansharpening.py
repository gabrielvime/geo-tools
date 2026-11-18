import numpy as np
from scipy.ndimage import zoom
import cv2

from type_aliases import ImageHW, ImageCHW, ImageHWC, ImageT

def nndiffuse(ms_img, pan_img, ratio, sigma_intensity=15.0, sigma_spatial=None):
    """
    Nearest-Neighbor Diffusion-Based Pan-Sharpening (NNDiffuse) in NumPy.
    
    Parameters:
    ms_img : ndarray of shape (H_low, W_low, Bands)
        Low-resolution multispectral image.
    pan_img : ndarray of shape (H_high, W_high)
        High-resolution panchromatic image.
    ratio : int
        The integer pixel size ratio between low and high resolution (e.g., 2, 4).
    sigma_intensity : float
        Intensity smoothness factor (σ). Controls similarity sensitivity.
    sigma_spatial : float, optional
        Spatial smoothness factor (σs). Defaults to ratio * 0.62.
        
    Returns:
    --------
    fused_img : ndarray of shape (H_high, W_high, Bands)
        High-resolution pan-sharpened image.
    """
    if sigma_spatial is None:
        sigma_spatial = ratio * 0.62  # Standard ENVI baseline

        if sigma_intensity is None or sigma_intensity == 15.0:
            sigma_intensity = np.std(pan_img) * 0.5 
            if sigma_intensity < 1.0: 
                sigma_intensity = 15.0
        
    h_high, w_high = pan_img.shape
    bands = ms_img.shape[2]
    
    # 1. Upsample MS image to match the PAN matrix dimensions via nearest neighbor
    # (Replicates the structural neighbor blocks/superpixels layout)
    ms_upsampled = zoom(ms_img, (ratio, ratio, 1), order=0)
    
    # Crop or pad to match the precise PAN dimensions if edge rounding happens
    ms_upsampled = ms_upsampled[:h_high, :w_high, :]
    
    # 2. Derive a low-resolution representation of PAN to compute delta shifts
    # Downsample via local block averaging, then upsample back smoothly
    pan_low = zoom(zoom(pan_img, 1/ratio, order=1), ratio, order=1)[:h_high, :w_high]
    pan_diff = pan_img - pan_low

    # 3. Calculate neighborhood diffusion weights using an 8-neighbor window (plus center)
    offsets = [-1, 0, 1]
    weight_stack = []
    spectrum_stack = []
    
    for di in offsets:
        for dj in offsets:
            # Shift the underlying structural array to vectorise neighbor calculation
            pan_shifted = np.roll(np.roll(pan_img, shift=-di, axis=0), shift=-dj, axis=1)
            ms_shifted = np.roll(np.roll(ms_upsampled, shift=-di, axis=0), shift=-dj, axis=1)
            
            # Spatial distance
            dist_sq = di**2 + dj**2
            
            # Diffusion equation weight: similarity * closeness
            spatial_term = np.exp(-dist_sq / (2 * (sigma_spatial ** 2)))
            intensity_term = np.exp(-((pan_img - pan_shifted) ** 2) / (2 * (sigma_intensity ** 2)))
            
            weight = spatial_term * intensity_term
            
            weight_stack.append(weight)
            spectrum_stack.append(ms_shifted)
            
    # Convert lists to high-dimensional tensors for seamless vectorization
    weights = np.stack(weight_stack, axis=-1)  # Shape: (H, W, 9)
    spectra = np.stack(spectrum_stack, axis=-2) # Shape: (H, W, 9, Bands)
    
    # Normalize the local weights across the 9 neighbor nodes
    weight_sum = np.sum(weights, axis=-1, keepdims=True)
    normalized_weights = weights / (weight_sum + 1e-8)
    
    # 4. Mix the local spatial features back into the multi-band spectrum
    # Dynamic blending utilizing broadcasting matrix rules
    fused_base = np.sum(spectra * normalized_weights[..., np.newaxis], axis=-2)
    #fused_base = np.sum(np.dot(spectra, normalized_weights[..., np.newaxis]), axis=-2)
    
    # Inject high-frequency detail scaling factor from the Panchromatic channel
    fused_img = np.zeros_like(ms_upsampled, dtype=np.float32)
    for b in range(bands):
        fused_img[..., b] = fused_base[..., b] + pan_diff
        
    return np.clip(fused_img, 0, np.max(ms_img))



import numpy as np
import cv2

def gsa(
    pan: np.ndarray[tuple[int, int], np.dtype[np.float32]], 
    ms: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    param: dict[str, np.float32] | None = None,
    clip_max:np.dtype[np.float32] | None = None
    ) -> tuple[np.ndarray[tuple[int, int, int], np.dtype[np.float32]], dict[str, np.float32]]:
    """Applies Gram-Schmidt Adaptive (GSA) for Pan-Sharpening.     

    Receives a Multispectral imagery and a pan imagery, uses the GSA algothim to pansharpen,
    if params are provided uses them, else calculate each one.
    returms the sharpened imagery and the parameters for reusability.
    
    Args:
        pan:   
            pan imagery in [H, W] format (no chanel).
        ms:
            multispectral imagery in [H, W, C] format (chanel last).
        param:
            paremeters of the regression, if provided they will be used, else them will be calculate.
    
    Returns:
        A tuple cointaing the pansharpened imagery and the parameters .

    Raises:
        ValueError: if the bands are not in the correct shape.


    """

    if pan.ndim != 2: raise ValueError(f"Incorrect {pan.shape} PAN shape, must be (H, W).")
    if ms.ndim != 3: raise ValueError(f"Incorrect {ms.shape} MS shape, must be (H, W , C).")

    H, W = pan.shape
    h, w, num_bands = ms.shape

    # MS upsampling using cubic interpolation
    if (h, w) != (H, W):
        ms_up = cv2.resize(ms, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        ms_up = ms.copy()

    ms_flat = ms_up.reshape(-1, num_bands)
    pan_flat = pan.ravel()

    # add 1s for linear equation bias
    X = np.hstack([ms_flat, np.ones((ms_flat.shape[0], 1))])

    if param is None:

        print(f'calculating params...')
        # regression: W = (X^T X)^-1 X^T Y
        weights, _, _, _ = np.linalg.lstsq(X, pan_flat, rcond=None)
        pan_sim_flat = X @ weights

        # Histogram Matching params
        mean_sim, std_sim = np.mean(pan_sim_flat), np.std(pan_sim_flat)
        mean_pan, std_pan = np.mean(pan_flat), np.std(pan_flat)
        if std_pan == 0: std_pan = 1.0 

        # (g_i) injection weights
        var_pan_sim = np.var(pan_sim_flat)
        g_i_array = np.zeros(num_bands)
        
        for i in range(num_bands):
            cov_matrix = np.cov(ms_flat[:, i], pan_sim_flat)
            cov_val = cov_matrix[0, 1]
            g_i = cov_val / var_pan_sim if var_pan_sim != 0 else 0
            g_i_array[i] = np.clip(g_i, -3.0, 3.0) # Atenuação para evitar tom rosa

        param = {
            'weights': weights,
            'mean_sim': mean_sim,
            'std_sim': std_sim,
            'mean_pan': mean_pan,
            'std_pan': std_pan,
            'g_i': g_i_array
        }
    else:
        print(f'using provided params...')
        # loads param values
        weights = param['weights']
        mean_sim = param['mean_sim']
        std_sim = param['std_sim']
        mean_pan = param['mean_pan']
        std_pan = param['std_pan']
        g_i_array = param['g_i']
        
        # simulated pan
        pan_sim_flat = X @ weights



    # Histogram Matching in original PAN
    pan_matched = ((pan_flat - mean_pan) / std_pan) * std_sim + mean_sim

    # details extraction
    high_freq_details = pan_matched - pan_sim_flat

    # injecting in ms bands
    ms_sharpened_flat = np.zeros_like(ms_flat, dtype=np.float32)
    for i in range(num_bands):
        ms_sharpened_flat[:, i] = ms_flat[:, i] + (g_i_array[i] * high_freq_details)

    # clipping to avoid underflow / overflow
    if clip_max is None:
        clip_max = np.max(ms)
        
    ms_sharpened_flat = np.clip(ms_sharpened_flat, 0, clip_max)

    ms_sharpened = ms_sharpened_flat.reshape(H, W, num_bands)

    return ms_sharpened, param



def gram_schmidt(
    pan: ImageHW,
    ms: ImageHWC,
    weights: ImageT | None = None,
    clip_range: tuple[float, float] | None = None,
) -> ImageHWC:
    """Perform classical Gram-Schmidt (GS) pan-sharpening.

    Simulates a low-resolution panchromatic band from the multispectral bands,
    applies forward Gram-Schmidt orthogonalization, replaces the first
    component with the histogram-matched high-resolution panchromatic band,
    and performs the inverse transformation.

    Args:
        pan: High-resolution panchromatic image with shape (H, W).
        ms: Low-resolution multispectral image with shape (h, w, bands).
        weights: Optional spectral weights of shape (bands,) to simulate the
            low-resolution PAN. Defaults to equal weights (1 / bands).
        clip_range: Optional (min, max) tuple to clip output values.

    Returns:
        Pan-sharpened multispectral image with shape (H, W, bands) and dtype float32.

    Raises:
        ValueError: If input array dimensions or weight shapes are invalid.
    """
    if pan.ndim != 2:
        raise ValueError(f"Expected 2D PAN array (H, W), got shape {pan.shape}.")
    if ms.ndim != 3:
        raise ValueError(f"Expected 3D MS array (h, w, bands), got shape {ms.shape}.")

    height, width = pan.shape
    ms_h, ms_w, num_bands = ms.shape

    if weights is None:
        weights = np.full(num_bands, 1.0 / num_bands, dtype=np.float32)
    elif weights.shape != (num_bands,):
        raise ValueError(f"Expected weights of shape ({num_bands},), got {weights.shape}.")

    if (ms_h, ms_w) != (height, width):
        ms_up = cv2.resize(ms, (width, height), interpolation=cv2.INTER_CUBIC)
        if ms_up.ndim == 2:
            ms_up = ms_up[:, :, np.newaxis]
    else:
        ms_up = ms.astype(np.float32, copy=False)

    num_pixels = height * width
    ms_flat = ms_up.reshape(num_pixels, num_bands).astype(np.float32, copy=False)
    pan_flat = pan.reshape(num_pixels).astype(np.float32, copy=False)

    # Simulate low-resolution PAN via weighted average (GS Mode 1)
    pan_sim = ms_flat @ weights.astype(np.float32, copy=False)

    # Forward Gram-Schmidt orthogonalization on mean-centered components
    gs = np.empty((num_pixels, num_bands + 1), dtype=np.float32)
    phi = np.zeros((num_bands + 1, num_bands + 1), dtype=np.float32)

    sim_mean = float(np.mean(pan_sim, dtype=np.float64))
    gs[:, 0] = pan_sim - sim_mean
    ms_means = np.mean(ms_flat, axis=0, dtype=np.float64).astype(np.float32)

    for b in range(1, num_bands + 1):
        gs[:, b] = ms_flat[:, b - 1] - ms_means[b - 1]
        for j in range(b):
            denom = float(np.dot(gs[:, j], gs[:, j]))
            if denom > 0.0:
                phi[b, j] = float(np.dot(gs[:, b], gs[:, j])) / denom
                gs[:, b] -= phi[b, j] * gs[:, j]

    # Substitute first component with histogram-matched high-resolution PAN
    pan_mean = float(np.mean(pan_flat, dtype=np.float64))
    pan_std = float(np.std(pan_flat, dtype=np.float64))
    sim_std = float(np.std(gs[:, 0], dtype=np.float64))

    if pan_std > 0.0:
        gs[:, 0] = ((pan_flat - pan_mean) / pan_std) * sim_std
    else:
        gs[:, 0] = 0.0

    # Inverse Gram-Schmidt transformation
    sharpened_flat = np.empty((num_pixels, num_bands), dtype=np.float32)
    for b in range(1, num_bands + 1):
        band = gs[:, b] + ms_means[b - 1]
        for j in range(b):
            band += phi[b, j] * gs[:, j]
        sharpened_flat[:, b - 1] = band

    if clip_range is not None:
        np.clip(sharpened_flat, clip_range[0], clip_range[1], out=sharpened_flat)

    return sharpened_flat.reshape(height, width, num_bands)