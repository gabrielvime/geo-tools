import numpy as np

def center_crop(array:np.array, crop_height:int, crop_width:int) -> np.array:
    """
    Center crop a Numpy Array .
    
    Parameters:
    -----------
    array : numpy array of shape (H_low, W_low, Bands)
        array to be cropped.
    crop_height : int
        height to crop.
    crop_width : int
        width to crop.
    
        
    Returns:
    --------
    cropped_array : ndarray of shape (H_high, W_high, Bands)
        croppped array.
    """
    try:
        h,w = array.shape[:2]

        start_x = (w - crop_width) // 2
        start_y = (h - crop_height) // 2

        return array[start_y:start_y + crop_height, start_x:start_x + crop_width]
    
    except Exception as e:
        print(e)