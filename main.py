#import stac, shape_tools, config, web_tiles
import imagery.stac, shape.shape_tools, raster.nnd_pansharpening, config
import geopandas as gpd
import os
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import rasterio
from datetime import date, timedelta, datetime

def main(source, test=False):

    if test:
        print(f'\n TEST MODE \n')
        ##################################################################################
        
            
        ##############################################################################
        exit()


    #### BUILDING MINING DATASET FOR CBERS, SENTINEL AND LANDSAT ####

    

    # list shapes
    shapes_list = os.listdir(config.SHAPE_PATH)

    
    # step 2 verirfy if there is a valid sentinel 2 imagery fpr the same cbers time window
    # step 3 do the same for landsat
    # step 4 correct spacial desalign (AROSICS)
    # step 5 calculate spctral indexes and stack in new geotiff
    # step 6 save in the format ID_REF-DATE_POS/NEG
    # step 7 save in table

    for geometry in shapes_list:

        # get geometry
        gdf = shape.shape_tools.gdf(config.SHAPE_PATH + '/' + geometry)     

        # iterate over geometries
        for idx, row in gdf.iterrows():
            print(f'\n {idx} - id: {row['id']} - processo: {row['processo']} \n')

            polygon = row['geometry'].envelope

            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)

            shapefile_name = f'{row['id']}'
            print(f'name: {shapefile_name}')

        
            # create directories
    
            output_dir = Path(config.SOURCES.get('cbers')[1])
            output_dir.mkdir(parents=True, exist_ok=True)

            # ## get imagery
            # imagery.stac.getData(
            #     gdf=data_frame, 
            #     shapefile_name=shapefile_name, 
            #     source='cbers', 
            #     timeframe=config.DATETIME, 
            #     size=128, 
            #     bands=['data'],
            #     #bands=['BAND1', 'BAND2', 'BAND3', 'BAND4'], 
            #     verify=True, 
            #     aux_shape=None, 
            #     getAll=False
            #     )

            # print(f'RGB got')

            # exit()

            # # imagery.stac.getData(gdf=data_frame, shapefile_name=shapefile_name + '_PAN', source='cbers', timeframe=config.DATETIME, size=512, bands=['BAND0'], verify=False, aux_shape=None, getAll=False)
            # # print(f'PAN got')

            # # NND Pansharpenning
            
            
            # with rasterio.open('cbers_imagery/131386_PAN_CBERS_4A_WPM_20260630_217_121_L4.tif') as src_pan:
                
                
            #     df_utm = data_frame.to_crs(src_pan.crs)
                
                
            #     window_pan = shape.shape_tools.window(df_utm, src_pan, max_size=config.MAX_SIZE)
            #     transform_pan = rasterio.windows.transform(window_pan, src_pan.transform)
            #     pan_crs = src_pan.crs
                
                
            #     pan_img = src_pan.read(1, window=window_pan, boundless=True, fill_value=0)
            #     bounds = rasterio.windows.bounds(window_pan, src_pan.transform)

            
            # with rasterio.open('cbers_imagery/131386_CBERS_4A_WPM_20260630_217_121_L4.tif') as src_ms:
            #     window_ms = rasterio.windows.from_bounds(*bounds, transform=src_ms.transform)
            #     low_res_img = src_ms.read(window=window_ms, boundless=True, fill_value=0)

            
            # low_res_img = np.transpose(low_res_img, (1,2,0))

            # low_res_img = np.ascontiguousarray(low_res_img, dtype=np.float32)
            # pan_img = np.ascontiguousarray(pan_img, dtype=np.float32)
            
            # fused_image = raster.nnd_pansharpening.pansharpen(
            #     ms_img=low_res_img,
            #     pan_img=pan_img,
            #     ratio=4
            #     )
            
            # fused_image = np.transpose(fused_image, (2, 0, 1))
            # fused_image = np.ascontiguousarray(fused_image, dtype=np.int16)
            
            # fused_meta = ({
            #     "driver": "GTiff",
            #     "count": 4,
            #     "height": fused_image.shape[1],
            #     "width": fused_image.shape[2],
            #     "crs": pan_crs,
            #     "transform": transform_pan,
            #     "dtype": fused_image.dtype.name,
            #     "nodata": 0
            # })

            
            
            # fused_name = 'ID_IMAGE_DATE_CBERS-4A-L4_POS.tiff'
            # with rasterio.open(fused_name, "w", **fused_meta) as dest:
            #     dest.write(fused_image)



            # # SEARCH FOR SENTINEL IMAGERY IN DATE FRAME

            # correct_name = '131386_PAN_CBERS_4A_WPM_20260630_217_121_L4'

            # date_str = correct_name.split('_')[5]
            # date_frame = datetime.strptime(date_str, "%Y%m%d").date()
            # date_upper = date_frame + timedelta(days=10)
            # date_lower = date_frame - timedelta(days=10)
            
            # date_frame = str(date_lower) + '/' + str(date_upper)

            # print(f'\ndate frame: {date_frame}\n')

            
            # # SENTINEL 2 10m BANDS

            # output_dir = Path(config.SOURCES.get('sentinel2')[1])
            # output_dir.mkdir(parents=True, exist_ok=True)

            # imagery.stac.getData(
            #     gdf=data_frame, 
            #     shapefile_name=shapefile_name, 
            #     source='sentinel2', 
            #     timeframe=date_frame, 
            #     size=128, 
            #     bands=['blue', 'green', 'red', 'nir'],
            #     verify=False,
            #     aux_shape=None, 
            #     getAll=False
            #     )

            with rasterio.open('cbers_imagery/131386_PAN_CBERS_4A_WPM_20260630_217_121_L4.tif') as src:
                
                
                df_utm = data_frame.to_crs(src_pan.crs)
                
                
                window_pan = shape.shape_tools.window(df_utm, src_pan, max_size=config.MAX_SIZE)
                transform_pan = rasterio.windows.transform(window_pan, src_pan.transform)
                pan_crs = src_pan.crs
                
                
                pan_img = src_pan.read(1, window=window_pan, boundless=True, fill_value=0)
                bounds = rasterio.windows.bounds(window_pan, src_pan.transform)


            














    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get sattellite imagery from desired source')

    parser.add_argument('--source', type=str, required=True, help='cbers, google, sentinel2...')
    parser.add_argument('--test', type=bool, required=False, help='test mode')

    args = parser.parse_args()

    main(source=args.source, test=args.test)

    