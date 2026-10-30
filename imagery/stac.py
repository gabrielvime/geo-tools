from pathlib import Path
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box, shape
import pandas as pd

import raster.validate, shape.shape_tools, config
from raster import interpolate

'''
Get imagery from a shape with optional polygon border overlay
'''

def getData(gdf, shapefile_name, source, timeframe=config.DATETIME, size=config.RESOLUTION, assets=[], verify=True, aux_shape=None, getAll=True):

    print(f'loading polygon...')
    
    bbox = gdf.total_bounds
    print(f'shape: {shapefile_name}')

    print(f'connecting to source...')

    if source =='cbers':
        items = config.cbers(bbox=bbox, datetime=timeframe)
    elif source == 'sentinel2':
        items = config.sentinel2(bbox=bbox, datetime=timeframe)
    elif source == 'landsat':
        items = config.landsdat(bbox=bbox, datetime=timeframe)
    
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


    success = False
    print(f'total scenes received: {len(items)}')
    print(f'processing fromm newest to oldest...')    
    items.sort(key=lambda x: x.datetime, reverse=True)
    
    for item in items:

        stacked_bands = []
        
        window = None

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

                with rasterio.open(band_href) as src:
                    
                    if window is None:
                        gdf = gdf.to_crs(src.crs)
                        scene_crs = src.crs
                        print(f'calculating scene bounds...')
                        
                        window = shape.shape_tools.window(gdf, src, max_size=config.MAX_SIZE, size=size, crs='EPSG:4326')
                        transform = rasterio.windows.transform(window, src.transform)

                    window = shape.shape_tools.window(gdf, src, max_size=config.MAX_SIZE, size=size, crs='EPSG:4326')
                    
                    # get image
                    # we fill missing values with numpy nan
                    print(f'getting band {band_key}...')
                    band_array = src.read(1,window=window, boundless=True, fill_value=-999)

                    print(f'-999 VALUES BEFORE: {np.count_nonzero(band_array == -999)}')
                    
                    # convert to np.float32
                    band_array = band_array.astype(np.float32)

                    # fill -999 no_data values with np.nan
                    band_array = np.where(band_array == -999, np.nan, band_array)
                    if src.nodata is not None:
                        band_array[band_array == src.nodata] = np.nan
                    
                    print(f'NAN VALUES BEFORE: {np.count_nonzero(np.isnan(band_array))}')

                    if np.isnan(band_array).any():
                        print(f'interpolating...')
                        band_array = interpolate.fmm(band_array, grid=5)

                    print(f'NAN VALUES: {np.count_nonzero(np.isnan(band_array))}')

                    stacked_bands.append(band_array)
                    
                    #print(f'getting band {band_key}... DONE!', end='', flush=True)
            
            image = np.stack(stacked_bands)

            if not config.MAX_SIZE:
                print(f'croping image...')
                image = np.ascontiguousarray(image[:, :size, :size])
                

            out_meta = ({
                "driver": "GTiff",
                "count": len(assets),
                "height": image.shape[1],
                "width": image.shape[2],
                "transform": transform,
                "crs": scene_crs,
                "dtype": image.dtype.name,
                "nodata": 0
            })
            
            #out_meta.pop("blockxsize", None)
            #ut_meta.pop("blockysize", None)
                    
            
            # image validations
            
            if verify:
                print(f'checking cloud cover...')
                if not raster.validate.cloud_filter(image, cloud_threshold=0.20, shadow_threshold=0.30) and not getAll:
                    print(f'cloud covered...')
                    print(f'skipping...')
                    continue

                if not raster.validate.contrast_filter(image=image, threshold=0.1) and not getAll:
                    print('contrast filter')
                    print('skipping...')
                    continue

                if not raster.validate.sharpness_filter(image=image, threshold=0.1) and not getAll:
                    print('sharpness filter')
                    print('skipping...')
                    continue
            
            print(f'verifing data integrity...')
            if not raster.validate.dataIntegrity(image, threshold=0.05) and not getAll:
                print(f'scene integrity compromissed..')
                print(f'skipping...')
                continue


            # draw polygon
            if config.DRAW_POLYGON:
                print(f'drawing polygon...')
                
                if aux_shape is None:
                    image = shape.shape_tools.draw(
                        image, 
                        shape=gdf, 
                        transform=transform,
                        crs=gdf.crs
                        )
                    
                else:
                    image = shape.shape_tools.draw(
                        image, 
                        shape=aux_shape, 
                        transform=transform,
                        crs=gdf.crs
                        )    
                
            # file saving
            print(f'saving file...')
            print(item.datetime)



            return (image, out_meta, metadata)
            

            # output_filename = config.SOURCES.get(source)[1] + '/' + f"{shapefile_name}_{item.id}.tif"
            # with rasterio.open(output_filename, "w", **out_meta) as dest:
            #     dest.write(image)

            # print(f'saved as {output_filename} with scene {item.id}')
            # print()

            success = True


            if success and getAll == False:
                break
            print()
        
                
        except Exception as e:
            print(f"failed in {item.id}: {e}")
            continue


    if not success:
        print("error: no scene found")
