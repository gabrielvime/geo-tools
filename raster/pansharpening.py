import numpy as np
from scipy.ndimage import zoom
import cv2

def nndiffuse(ms_img, pan_img, ratio, sigma_intensity=15.0, sigma_spatial=None):
    """
    Nearest-Neighbor Diffusion-Based Pan-Sharpening (NNDiffuse) in NumPy.
    
    Parameters:
    -----------
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



def gsa(pan: np.ndarray, ms: np.ndarray, weights=None) -> np.ndarray:
    """
    Applies the Gram-Schmidt Adaptive (GSA) algorithm for Pan-Sharpening.

    Args:
        pan (np.ndarray): High-resolution panchromatic image (2D array, shape: H x W).
        ms (np.ndarray): Low-resolution multispectral image (3D array, shape: h x w x Bands).

    Returns:
        np.ndarray: High-resolution multispectral image (Pan-sharpened).
    """
    # Check image dimensions
    if pan.ndim != 2:
        raise ValueError("The PAN image must have 2 dimensions (H, W).")
    if ms.ndim != 3:
        raise ValueError("The MS image must have 3 dimensions (h, w, Bands).")

    H, W = pan.shape
    h, w, num_bands = ms.shape

    # 1. Upsampling the MS image to match PAN size (using Bicubic interpolation)
    if (h, w) != (H, W):
        ms_up = cv2.resize(ms, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        ms_up = ms.copy()

    # Flatten matrices for easier statistical and matrix calculations
    ms_flat = ms_up.reshape(-1, num_bands)
    pan_flat = pan.ravel()

    # 2. Simulation of the PAN band (Pan_Sim) via Multiple Linear Regression
    # Add a column of 1s for the bias (intercept) of the linear equation
    X = np.hstack([ms_flat, np.ones((ms_flat.shape[0], 1))])
    
    if weights is None:
        # Solve the regression: W = (X^T X)^-1 X^T Y
        print(f'erro aqui')
        exit()
        weights, _, _, _ = np.linalg.lstsq(X, pan_flat, rcond=None)


    # Generate simulated Pan using calculated weights
    pan_sim_flat = X @ weights
    pan_sim = pan_sim_flat.reshape(H, W)

    # 3. Extraction of high-frequency details (Original Pan - Simulated Pan)
    high_freq_details = pan - pan_sim

    # 4. Calculation of injection weights (g) and application to the MS image
    ms_sharpened = np.zeros_like(ms_up, dtype=np.float32)
    var_pan_sim = np.var(pan_sim_flat)

    for i in range(num_bands):
        # g_i = covariance(Band_i, Pan_sim) / variance(Pan_sim)
        cov_matrix = np.cov(ms_flat[:, i], pan_sim_flat)
        cov_val = cov_matrix[0, 1]
        
        # Prevent division by zero
        g_i = cov_val / var_pan_sim if var_pan_sim != 0 else 0
        
        # 5. Inject details into the current band
        ms_sharpened[:, :, i] = ms_up[:, :, i] + (g_i * high_freq_details)

    return ms_sharpened