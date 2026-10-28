import numpy as np

# Inverse Normalized Difference Vegetation Index
def indvi(nir:np.float32, red:np.float32) -> np.float32:
    denom = nir + red
    return 1 * (np.where(denom == 0, 0, (nir - red) / denom))

# Green Normalized Difference Vegetation Index
def gndvi(green:np.float32, nir:np.float32) -> np.float32:
    denom = nir + green
    return np.where(denom == 0, 0, (nir - green) / denom)

# Normalized Difference Turbidity Index 
def ndti(red:np.float32, green:np.float32) -> np.float32:
    denom = red + green
    return np.where(denom == 0, 0, (red - green) / denom)

# Normalized Difference Water Index
def ndwi(green:np.float32, nir:np.float32) -> np.float32:
    denom = green + nir
    return np.where(denom == 0, 0, (green - nir) / denom)

# Modified Normalized Difference Water Index 
def mndwi(green, swir1):
    denom = green + swir1
    return np.where(denom == 0, 0, (green - swir1) / denom)

# Modified Bare Soil Index
def mbi(swir1, swir2, nir, factor):
    denom = swir1 + swir2 + nir
    return np.where(denom == 0, 0, ( (swir1 - swir2 - swir2) / denom ) + factor)

# Normalized Difference Red Edge Index 
def ndre(nir, red_edge):
    denom = nir + red_edge
    return np.where(denom == 0, 0, (nir - red_edge) / denom)

# Land Surface Temperature (Landsat 8)
def lst(dn:np.float32) -> np.float32:
    return (dn * 0.00341802) + 149
