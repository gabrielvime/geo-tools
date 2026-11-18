from pathlib import Path
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import box, shape
import pandas as pd
import time
import requests

import shape.shape_tools, config
from raster import interpolate, validate, array_tools

'''
Get imagery from a shape with optional polygon border overlay
'''

def getData(
        gdf, shapefile_name, 
        source, timeframe=config.DATETIME, 
        size=config.RESOLUTION, 
        max_size=config.MAX_SIZE,
        assets=[], 
        verify=True, 
        dec_read=1, 
        getAll=True
    ):

    print(f'loading polygon...')
    
    bbox = gdf.total_bounds
    print(f'shape: {shapefile_name}')

    print(f'connecting to source...')


    for attemp in range(5):
        try:
            if source =='cbers':
                items = config.cbers(bbox=bbox, datetime=timeframe)
            elif source == 'sentinel2':
                items = config.sentinel2(bbox=bbox, datetime=timeframe)
            elif source == 'landsat':
                items = config.landsdat(bbox=bbox, datetime=timeframe)
        except Exception as e:
            print(f'Server error (attemp {attemp + 1}/5): {e}')

            if attemp < 10:
                for i in range(60, 0, -1):
                    print(f'Waiting {i} seconds to try again...', end='\r', flush=True)
                    time.sleep(1)
                    continue
            else:
                raise
        
    if not items:
        print(f'no scenes obtained from given parameters')
        return

    print()
    print(f'getting scenes from:')
    print(f'SOURCE: {config.SOURCES.get(source)[0]}')
    print(f'COLLECTION: {items[0].collection_id}')
    print(f'DATE FRAME:{items[len(items) - 1].datetime} to {items[0].datetime}')
    print(f'SHAPE: {shapefile_name}')
    print()

    #print(f'\n max size = {max_size}')

    success = False
    print(f'total scenes received: {len(items)}')
    print(f'processing fromm newest to oldest...')    
    items.sort(key=lambda x: x.datetime, reverse=True)

    no_data_fill = -999
    
    for item in items:

        stacked_bands = []
        
        window = None

        print(f'working in {item.datetime.date()}:\n')
        metadata = {
                    'DATE': item.datetime.date(), 
                    'NAME': f'{item.id}',
                    'SIZE': size
                    }

        try:

            for band_key in assets:

                if band_key not in item.assets:
                    raise ValueError(f'band {band_key} not found')

                band_href = item.assets[band_key].href

                response = requests.head(band_href, allow_redirects=True, timeout=10)
                size = int(response.headers.get('Content-Length', 0))

                

                with rasterio.open(band_href) as scene_data:
                    
                    if window is None:
                        gdf = gdf.to_crs(scene_data.crs)
                        scene_crs = scene_data.crs
                        print(f'calculating scene bounds... ', end="", flush=True)
                        
                        window = shape.shape_tools.window(gdf, scene_data, max_size=max_size, size=size, crs='EPSG:4326')
                        transform = rasterio.windows.transform(window, scene_data.transform)
                        print(f' DONE')

                    window = shape.shape_tools.window(gdf, scene_data, max_size=max_size, size=size, crs='EPSG:4326')
                    
                    # get band data
                    # generally comes as np.ndarray(nd.dtype(np.int16))

                    print_getting_band = f'getting band {band_key}... '
                    print(f'\r{print_getting_band} (Size to download: {size / (1024**2):.2f} MB)\033[K', end="", flush=True)

                    if dec_read > 1:
                        dec_height = int(window.height // dec_read) 
                        dec_width = int(window.width // dec_read)

                        band_array = scene_data.read(
                            1, 
                            window=window, 
                            boundless=True, 
                            fill_value=no_data_fill, 
                            out_shape=(dec_height, dec_width),
                            resampling=Resampling.bilinear
                            ).astype(np.float32)
                    else:


                        band_array = scene_data.read(
                            1, 
                            window=window, 
                            boundless=True, 
                            fill_value=no_data_fill
                            ).astype(np.float32)

                    #band_array = band_array.astype(np.float32)

                    # crops the imagery if dont want max_size
                    # max_size is intended for calculate global params
                    # or simply get thee whole imagery
                    if not max_size:

                        print(f'\r{print_getting_band} (croping image)\033[K', end="", flush=True)
                        band_array = np.ascontiguousarray(array_tools.center_crop(band_array, size, size))
                    
                    print(f'\r{print_getting_band} (verifing data integrity)\033[K', end="", flush=True) 

                    # calculates the no_data values
                    compromissed_integrity = validate.dataIntegrity(band_array, no_data=no_data_fill)

                    # if data integrity is compromissed but under threshold
                    # interpolates the no_data
                    # if is bigger just skips
                    # if is to getAll, then just save corrupted imagery
                    if (compromissed_integrity > 0.0) and not getAll:

                        if compromissed_integrity < 0.005:
                            print(f'\r{print_getting_band} (interpolating remaing invalid data)\033[K', end="", flush=True)
                            band_array = interpolate.fmm(image=band_array, grid=5, no_data=no_data_fill)

                        else:
                            print(f'\nscene integrity compromissed in {compromissed_integrity}...')
                            print(f'skipping...\n')

                            raise ValueError(f'Corrupted data.')          
                        
                    print(f"\r{print_getting_band} (DONE)\033[K")

                    stacked_bands.append(band_array)
            
            image = np.stack(stacked_bands)
            image = np.transpose(image, (1,2,0))

            out_meta = ({
                "driver": "GTiff",
                "count": len(assets),
                "height": image.shape[0],
                "width": image.shape[1],
                "transform": transform,
                "crs": scene_crs,
                "dtype": image.dtype.name,
                "nodata": np.nan
            })
                    
            
            # image validations
            
            if verify:
                print(f'checking cloud cover...')
                if not validate.cloud_filter(image, cloud_threshold=0.20, shadow_threshold=0.30) and not getAll:
                    print(f'cloud covered...')
                    print(f'skipping...')
                    continue

                if not validate.contrast_filter(image=image, threshold=0.1) and not getAll:
                    print('contrast filter')
                    print('skipping...')
                    continue

                if not validate.sharpness_filter(image=image, threshold=0.1) and not getAll:
                    print('sharpness filter')
                    print('skipping...')
                    continue

            # draw polygon
            if config.DRAW_POLYGON:
                print(f'drawing polygon...')
            
                image = shape.shape_tools.draw(
                    image, 
                    shape=gdf, 
                    transform=transform,
                    crs=gdf.crs
                    )
  
                
            # file saving
            print(f'saving file...')
            return (image, out_meta, metadata)

            success = True


            if success and getAll == False:
                break
            print()
        
                
        except Exception as e:
            print(f"failed in {item.id}: {e}\n")
            continue

    if not success:
        print("error: no scene found")
