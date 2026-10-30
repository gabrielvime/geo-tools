#import stac, shape_tools, config, web_tiles
import config
from shape import shape_tools
from raster import indexes, pansharpening, normalize
from imagery import stac

import geopandas as gpd
import os
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import rasterio
from datetime import date, timedelta, datetime
from scipy import ndimage
import csv

def main(source, test=False):

    if test:
        print(f'\n TEST MODE \n')
        ##################################################################################
        
            
        ##############################################################################
        exit()


    #### BUILDING MINING DATASET FOR CBERS, SENTINEL AND LANDSAT ####

    

    # list shapes
    shapes_list = os.listdir(config.SHAPE_PATH)

    save_dir = './PA/CBERS'


    for geometry in shapes_list:

        # get geoDATAFRAME
        gdf = shape_tools.gdf(config.SHAPE_PATH + '/' + geometry)     

        # iterate over geometries
        for idx, row in gdf.iterrows():
            print(f'\n {idx} - id: {row['id']} - processo: {row['processo']} \n')

            polygon = row['geometry'].envelope

            csv_metadata = {
                'NOME': None,
                'ID': row['id'], 
                'PROCESSO': row['processo'], 
                'DATA': None, 
                'FASE': row['fase'], 
                'SUBSTANCIA': row['substancia'], 
                'UF': row['uf'], 
                'BIOMA': row['biomas'], 
                'USO': row['uso'],
                'PARTE': None,
                'SATELITE': source,
                'PANSHARPENING': None,
                'ASSETS': None,
                'TAMANHO': None
            }
            
            # CSV FILE HEADERS
            headers = list(csv_metadata.keys())

            print(headers)

            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)

            shapefile_name = f'{row['id']}'
            print(f'name: {shapefile_name}')

        
            # create directories
    
            output_dir = Path(config.SOURCES.get(source)[1])
            output_dir.mkdir(parents=True, exist_ok=True)

            ## get cbers imagery
            cbers_imagery, cbers_meta, _ = stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name, 
                source='cbers', 
                timeframe=config.DATETIME, 
                size=128, 
                assets=['BAND1', 'BAND2', 'BAND3', 'BAND4'], 
                verify=True, 
                aux_shape=None, 
                getAll=False
                )


            # exit()

            cbers_pan, cbers_pan_meta, cbers_outmeta = stac.getData(
                gdf=data_frame, 
                shapefile_name=shapefile_name + '_PAN', 
                source='cbers', 
                timeframe=config.DATETIME, 
                size=512, 
                assets=['BAND0'], 
                verify=False, 
                aux_shape=None, 
                getAll=False
                )

            print(f'CBERS PAN got')

            csv_metadata['DATA'] = cbers_outmeta['DATE']
            # csv_metadata['NOME'] = cbers_outmeta['NAME']
            csv_metadata['TAMANHO'] = cbers_outmeta['SIZE']
            print(f'CBERS IMAGERY GOT')
            # # NND Pansharpenning


            


            print(f'separate PAN band...')

            # cbers_pan = cbers_imagery[0]

            ## delete the pan band from the original stack
            # cbers_imagery = np.delete[cbers_imagery, 0]

            
            fusion = 'nnd'
            print(f'\n starting {fusion} pansharpening\n')
            low_res_img = np.transpose(cbers_imagery, (1,2,0))

            
            low_res_img = np.ascontiguousarray(low_res_img, dtype=np.float32)
            pan_img = np.ascontiguousarray(cbers_pan, dtype=np.float32)[0, :, :]
            
            
            fused_image = pansharpening.nndiffuse(
                ms_img=low_res_img,
                pan_img=pan_img,
                ratio=4
                )
            
            print('image fused!')
            fused_image = np.transpose(fused_image, (2, 0, 1))
            fused_image = np.ascontiguousarray(fused_image, dtype=np.float32)

            cbers_assets = ['red', 'green', 'blue', 'nir', 'ndvi', 'ndwi']

            csv_metadata['ASSETS'] = cbers_assets

            print('separating bands...')
            # BANDS
            blue = fused_image[0]
            green = fused_image[1]
            red = fused_image[2]
            nir = fused_image[3]
            print('cauculating indexes...')
            ndvi = indexes.ndvi(nir=nir, red=red)
            ndwi = indexes.ndwi(green=green, nir=nir)

            print(f'normalizing...')
            nomalized_red = normalize.winsorization_min_max(band=red)
            nomalized_green = normalize.winsorization_min_max(band=green)
            nomalized_blue = normalize.winsorization_min_max(band=blue)
            nomalized_nir = normalize.winsorization_min_max(band=nir)
            nomalized_ndvi = normalize.winsorization_min_max(band=ndvi)
            nomalized_ndwi = normalize.winsorization_min_max(band=ndwi)


            print('stacking...')
            fused_cbers = np.stack([
                nomalized_red, 
                nomalized_green, 
                nomalized_blue, 
                nomalized_nir, 
                nomalized_ndvi, 
                nomalized_ndwi
                ])

            print(f'NAN VALUES: {np.count_nonzero(np.isnan(fused_cbers))}')

            csv_metadata['PANSHARPENING'] = fusion
            
            fused_meta = cbers_pan_meta
            fused_meta.update({
                "count": min(fused_cbers.shape),
                "height": fused_image.shape[1],
                "width": fused_image.shape[2],
                "dtype": np.float32,
                "nodata": np.nan
            })
 
            
            fused_name = f'{source}_{row['id']}_{csv_metadata['PROCESSO'].replace('/', '-')}_{str(csv_metadata['DATA'])}_{fusion}_normalized.tiff'
            csv_metadata['NOME'] = fused_name

            print(csv_metadata)

            print('\n wiriting csv file')
            # WRITES DATA TO CSV
            with open(save_dir + '/metadata.csv', mode='w', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
                csv_writer.writeheader()
                csv_writer.writerow(csv_metadata)

            print(f'\nsaving geotiff as {fused_name}\n')
            # SAVE GEOTIFF FILE
            with rasterio.open(save_dir + '/UNCLASSIFIED/' + fused_name, "w", **fused_meta) as dest:
                dest.write(nomalized_red, 1)
                dest.set_band_description(1, 'red')
                
                dest.write(nomalized_green, 2)
                dest.set_band_description(2, 'green')

                dest.write(nomalized_blue, 3)
                dest.set_band_description(3, 'blue')

                dest.write(nomalized_nir, 4)
                dest.set_band_description(4, 'nir')

                dest.write(nomalized_ndvi, 5)
                dest.set_band_description(5, 'ndvi')

                dest.write(nomalized_ndwi, 6)
                dest.set_band_description(6, 'ndwi')

            print('ALL DONE')

           














    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get sattellite imagery from desired source')

    parser.add_argument('--source', type=str, required=True, help='cbers, google, sentinel2...')
    parser.add_argument('--test', type=bool, required=False, help='test mode')

    args = parser.parse_args()

    main(source=args.source, test=args.test)

    