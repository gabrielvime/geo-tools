import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box
from rasterio.windows import Window
import rasterio

import config
from type_aliases import ImageHW, ImageCHW, ImageHWC, ImageT

def center_crop(array:np.array, crop_height:int, crop_width:int) -> np.array:
    """
    Center crop a Numpy Array .
    
    Parameters:
        array : numpy array of shape (H_low, W_low)
            array to be cropped.
        crop_height : int
            height to crop.
        crop_width : int
            width to crop.
    
        
    Returns:
        cropped_array : ndarray of shape (H_high, W_high)
            croppped array.
    """
    try:
        h,w = array.shape

        start_x = (w - crop_width) // 2
        start_y = (h - crop_height) // 2

        return array[start_y:start_y + crop_height, start_x:start_x + crop_width]
    
    except Exception as e:
        print(e)

def divide(
    image : ImageT, 
    shape_file : gpd.GeoDataFrame,
    transform : rasterio.Affine,
    crs : str | rasterio.crs.CRS="EPSG:3857", 
    tile_size : int = config.RESOLUTION,
    overlap : int = int(config.RESOLUTION * 0.05), 
    spatial_resolution : float = 1
 ) -> list[ImageT]:
    """Divides an image into spatial tiles intersecting a target area.

    Image mode: Slices a 3D NumPy array (CHW or HWC) into sub-tiles intersecting
    the shapefile geometries, preserving the original channel layout.

    Args:
        image: Input 3D array formatted as CHW or HWC.
        shape_file: GeoDataFrame defining the target region of interest.
        transform: Affine transformation matrix mapping pixel coordinates to CRS.
        crs: Coordinate Reference System for spatial alignment. Defaults to "EPSG:3857".
        tile_size: Width and height of the output tiles in pixels. Defaults to config.RESOLUTION.
        overlap: Overlap between adjacent tiles in pixels. Defaults to 5% of config.RESOLUTION.
        spatial_resolution: Ground sampling distance (meters per pixel). Defaults to 1.0.

    Returns:
        A list of sliced NumPy arrays matching the input layout (`ImageT`) if `image`
        is provided, or a list of single-row GeoDataFrames if `image` is None.

    Raises:
        ValueError: If `overlap` >= `tile_size`, or if `image` is not a 3D array.
    """
    if overlap >= tile_size:
        raise ValueError("Overlap must be strictly less than tile_size.")

    if shape_file.crs != crs:
        shape_file = shape_file.to_crs(crs)

    target_geometry = unary_union(shape_file.geometry)

    if image is not None:
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (CHW or HWC), got shape {image.shape}.")

        # Infer layout: assume channels-first if axis 0 is strictly smaller than spatial axes
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        work_image = image if is_chw else np.transpose(image, (2, 0, 1))

        _, height, width = work_image.shape
        step = tile_size - overlap
        image_tiles: list[ImageT] = []

        for row in range(0, height - tile_size + 1, step):
            for col in range(0, width - tile_size + 1, step):
                window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)
                bounds = rasterio.windows.bounds(window, transform)
                tile_bbox = box(*bounds)

                if target_geometry.intersects(tile_bbox):
                    tile_chw = work_image[:, row : row + tile_size, col : col + tile_size]
                    out_tile = tile_chw if is_chw else np.transpose(tile_chw, (1, 2, 0))

                    tile_transform = rasterio.windows.transform(window, transform) 

                    image_tiles.append((out_tile, tile_transform))  # type: ignore[arg-type]

        return image_tiles