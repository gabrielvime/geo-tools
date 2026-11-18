from typing import NewType, TypeVar
import numpy as np

# BASE TYPES
Float32Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
Float32Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]

# NOMINAL TYPES
ImageHW = NewType("ImageHW", Float32Array2D)
ImageCHW = NewType("ImageCHW", Float32Array3D)  # Rasterio / PyTorch
ImageHWC = NewType("ImageHWC", Float32Array3D)  #  Matplotlib / OpenCV
ImageT = TypeVar("ImageT", ImageCHW, ImageHWC)
