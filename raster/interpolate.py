import numpy as np
import cv2

def fmm(image:np.float32, grid:int=3) -> np.array:
    """OpenCV's Fast Marching Method (FMM) inpaint.

    seamlessly blend the points with their immediate surrounding 'grid' x 'grid' neighborhoods.

    Args:
        image (np.float32): numpy array image with NaN valuees. ensure to be in float32 type.
        grid (int, optional): grid size. 
            Defaults to 3

    Returns:
        np.array: numpy array image with NaN values filled.

    Raises:
        ValueError: If the imagee is not in no.float32 format.
    """ 

    try:
        # binary mask of wherre the NaN valuees aree
        mask = np.isnan(image).astype(np.uint8) * 255

        # temporary set Nan to 0
        image[np.isnan(image)] = 0

        return cv2.inpaint(image, mask, inpaintRadius=grid, flags=cv2.INPAINT_TELEA)

    except Exception as e:
        print(e)
