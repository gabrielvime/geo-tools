#import stac, shape_tools, config, web_tiles
import imagery.stac, shape.shape_tools, raster.nnd_pansharpening, config
from raster import indexes
import geopandas as gpd
import os
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import rasterio
from datetime import date, timedelta, datetime
from scipy import ndimage

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

            ## get cbers imagery
            cbers_imagery, cbers_meta, date_frame = imagery.stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name, 
                source='cbers', 
                timeframe=config.DATETIME, 
                size=255, 
                assets=['BAND1', 'BAND2', 'BAND3', 'BAND4'], 
                verify=True, 
                aux_shape=None, 
                getAll=False
                )

            print(f'CBERS IMAGERY GOT')

            # exit()

            cbers_pan, cbers_pan_meta, _ = imagery.stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name + '_PAN', 
                source='cbers', 
                timeframe=config.DATETIME, 
                size=1020, 
                assets=['BAND0'], 
                verify=False, 
                aux_shape=None, 
                getAll=False
                )

            print(f'CBERS PAN got')

            # # NND Pansharpenning

            
            low_res_img = np.transpose(cbers_imagery, (1,2,0))

            
            low_res_img = np.ascontiguousarray(low_res_img, dtype=np.float32)
            pan_img = np.ascontiguousarray(cbers_pan, dtype=np.float32)[0, :, :]
            
            
            fused_image = raster.nnd_pansharpening.pansharpen(
                ms_img=low_res_img,
                pan_img=pan_img,
                ratio=4
                )
            
            fused_image = np.transpose(fused_image, (2, 0, 1))
            fused_image = np.ascontiguousarray(fused_image, dtype=np.int16)

            red = fused_image[2]
            green = fused_image[1]
            blue = fused_image[0]
            nir = fused_image[3]
            indvi = indexes.indvi(nir=nir, red=red)
            gndvi = indexes.gndvi(green=green, nir=nir)
            ndti = indexes.ndti(red=red, green=green)

            fused_cbers = np.stack([red, green, blue, nir, indvi, gndvi, ndti])
            
            fused_meta = cbers_pan_meta
            fused_meta.update({
                "count": min(fused_cbers.shape),
                "height": fused_image.shape[1],
                "width": fused_image.shape[2],
                "dtype": fused_image.dtype.name,
                "nodata": 0
            })

            
            
            fused_name = 'ID_IMAGE_DATE_CBERS-4A-L4_FUSED.tiff'
            with rasterio.open(fused_name, "w", **fused_meta) as dest:
                dest.write(red, 1)
                dest.write(green, 2)
                dest.write(blue, 3)
                dest.write(indvi, 4)
                dest.write(gndvi, 5)
                dest.write(ndti, 6)

                bands_names = ['red', 'green', 'blue', 'INDVI', 'GNDVI', 'NDTI']
                for idx, name in enumerate(bands_names, start=1):
                    dest.set_band_description(idx, name)


            # # SEARCH FOR SENTINEL IMAGERY IN DATE FRAME

            date_frame = date_frame.date()

            # date_str = str(date_frame).split(' ')[0]
            # date_frame = datetime.strptime(date_str, "%Y%m%d").date()
            date_upper = date_frame + timedelta(days=10)
            date_lower = date_frame - timedelta(days=10)
            
            date_frame = str(date_lower) + '/' + str(date_upper)

            print(f'\ndate frame: {date_frame}\n')

            
            # # SENTINEL 2 10m BANDS

            output_dir = Path(config.SOURCES.get('sentinel2')[1])
            output_dir.mkdir(parents=True, exist_ok=True)

            sentinel_imagery10, sentinel10_meta,_ = imagery.stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name, 
                source='sentinel2', 
                timeframe=date_frame, 
                size=204, 
                assets=['blue', 'green', 'red', 'nir'],
                verify=False,
                aux_shape=None, 
                getAll=False
                )

            # SENTINEL 2 20m BANDS

            output_dir = Path(config.SOURCES.get('sentinel2')[1])
            output_dir.mkdir(parents=True, exist_ok=True)

            shapefile_name = 'sentinel-2_20m'
            sentinel_imagery20, _, _ = imagery.stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name, 
                source='sentinel2', 
                timeframe=date_frame, 
                size=102, 
                assets=['rededge1', 'rededge2', 'rededge3', 'swir16', 'swir22'],
                verify=False,
                aux_shape=None, 
                getAll=False
                )

            
            resampling_factors = (1,2,2)

            resampled_sentinel20 = ndimage.zoom(sentinel_imagery20, zoom=resampling_factors, order=1)

            print(f' sentinel shape 10: {sentinel_imagery10.shape}')
            print(f' sentinel shape 20: {resampled_sentinel20.shape}')

            sentinel_imagery = np.concatenate([sentinel_imagery10, resampled_sentinel20], axis=0)


            sentinel_meta = sentinel10_meta
            sentinel_meta.update({
                "count": min(sentinel_imagery.shape),
                "height": sentinel_imagery.shape[1],
                "width": sentinel_imagery.shape[2],
                "dtype": sentinel_imagery.dtype.name,
                "nodata": 0
            })

            
            
            sentinel_name = 'SENTINEL2-RESAMPLED-INDEXES.tiff'
            with rasterio.open(sentinel_name, "w", **sentinel_meta) as dest:
                dest.write(sentinel_imagery)



            # Landsat 8 Thermal

            output_dir = Path(config.SOURCES.get('sentinel2')[1])
            output_dir.mkdir(parents=True, exist_ok=True)

            shapefile_name = 'landsat_lst'
            lst_imagery, meta, _ = imagery.stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name, 
                source='landsat', 
                timeframe=date_frame, 
                size=68, 
                assets=['lwir11'],
                verify=False,
                aux_shape=None, 
                getAll=False
                )

            # convert to float 32
            lst_imagery = lst_imagery.astype(np.float32)
            
            # calculate lst
            lst_imagery = indexes.lst(lst_imagery)

            # convert again to int16
            lst_imagery = lst_imagery.astype(np.int16)
            
            
            with rasterio.open('landsat_LST_scene', 'w', **meta) as dest:
                dest.write(lst_imagery)














    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get sattellite imagery from desired source')

    parser.add_argument('--source', type=str, required=True, help='cbers, google, sentinel2...')
    parser.add_argument('--test', type=bool, required=False, help='test mode')

    args = parser.parse_args()

    main(source=args.source, test=args.test)

    