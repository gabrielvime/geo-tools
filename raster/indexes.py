# Inverse Normalized Difference Vegetation Index
def indvi(nir, red):
    denom = nir + red
    return 1 - (np.where(denom == 0, 0, (nir - red) / denom))

# Green Normalized Difference Vegetation Index
def gndvi(green, nir):
    denom = nir + green
    return np.where(denom == 0, 0, (nir - green) / denom)

# Ferric Iron Index
def fii(red, green):
    return np.where(blue == 0, 0, red / green)  

# Iron Oxide Ratio
def ior(red, blue):
    return np.where(blue == 0, 0, red / blue) 

# Green-Blue Ratio (errous Iron Proxy)
def gbp(green, blue):
    return np.where(blue == 0, 0, green / blue)  

# Normalized Difference Turbidity Index 
def ndti(red, green):
    denom = red + green
    return np.where(denom == 0, 0, (red - green) / denom) 

# Canopy Structure Evaluation
def cse(nir, green):
    return np.where(blue == 0, 0, nir / green)  
