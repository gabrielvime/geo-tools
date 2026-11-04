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
import cv2

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

            # print(headers)

            gdf_list = []
            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)

            print(f'data frame = {data_frame.total_bounds}')

            for shape in shape_tools.divide(shape=data_frame, overlap=64, spatial_resolution=2):

                shape_gdf = gpd.GeoDataFrame(geometry=[shape], crs="EPSG:3857")
                shape_gdf = shape_gdf.to_crs(data_frame.crs)
                gdf_list.append(shape_gdf)

            # print(f'list size: {len(gdf_list)}')
            # exit()
        
            # create directories
    
            output_dir = Path(config.SOURCES.get(source)[1])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            idx=1
            for part in gdf_list:

                print(f'\n STARTING FOR {idx} OF {len(gdf_list)}\n')

                shapefile_name = f'{row['id']}'
                print(f'name: {shapefile_name}')
               
                # if idx != 16:
                #     idx += 1
                #     continue

                ## get RGB NIR cbers imagery
                cbers_imagery, cbers_meta, _ = stac.getData(
                    gdf=part   , 
                    shapefile_name=shapefile_name, 
                    source='cbers', 
                    timeframe=config.DATETIME, 
                    size=int(config.RESOLUTION // 4), 
                    assets=['BAND1', 'BAND2', 'BAND3', 'BAND4'], 
                    verify=False, 
                    aux_shape=None, 
                    getAll=False
                    )


                # exit()

                cbers_pan, cbers_pan_meta, cbers_outmeta = stac.getData(
                    gdf=part, 
                    shapefile_name=shapefile_name + '_PAN', 
                    source='cbers', 
                    timeframe=config.DATETIME, 
                    size=config.RESOLUTION, 
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


                print(f'separating PAN band...')

                # cbers_pan = cbers_imagery[0]

                # # delete the pan band from the original stack
                # cbers_imagery = np.delete[cbers_imagery, 0]

                # cut = ( cbers_imagery.shape[1] - ( (cbers_imagery.shape[1])/4 ) ) / 2
                
                # cbers_pan[cut:-cut, cut:-cut]


                nir = cbers_imagery[3]
                print('cauculating indexes...')
                ndvi = indexes.ndvi(nir=nir, red=cbers_imagery[2])
                ndwi = indexes.ndwi(green=cbers_imagery[1], nir=nir)

                # remove nir from stack to not mess with pansharpenning
                cbers_imagery = np.delete(cbers_imagery, 3, axis=0)


                fusion = 'gsa'
                cbers_imagery = np.transpose(cbers_imagery, (1,2,0))

                
                cbers_imagery = np.ascontiguousarray(cbers_imagery, dtype=np.float32)
                pan_img = np.ascontiguousarray(cbers_pan, dtype=np.float32)[0, :, :]
                
                ratio = 4

                if (cbers_imagery.shape[1] * ratio) < pan_img.shape[1]:

                    diff = pan_img.shape[1] - ( cbers_imagery.shape[1] * ratio )
                    pan_img = pan_img[:-diff,:-diff]
                
                print(f' PAN SHAPE: {pan_img.shape}')
                print(f' RGB SHAPE: {cbers_imagery.shape}')


                
                # # NND Pansharpenning
                print(f'\n starting {fusion} pansharpening\n')
                if fusion == 'nnd':
                    fused_image = pansharpening.nndiffuse(
                        ms_img=cbers_imagery,
                        pan_img=pan_img,
                        ratio=ratio
                        )
                elif fusion == 'gsa':
                    fused_image = pansharpening.gsa(pan=pan_img, ms=cbers_imagery, weights=np.array([-0.91362598,  2.87242526, -0.64374208, 67.68559085]))
                
                print('image fused!')
                fused_image = np.transpose(fused_image, (2, 0, 1))
                fused_image = np.ascontiguousarray(fused_image, dtype=np.float32)

                cbers_assets = ['red', 'green', 'blue', 'nir', 'ndvi', 'ndwi']

                csv_metadata['ASSETS'] = cbers_assets

                print('separating fusioned bands...')
                # BANDS
                blue = fused_image[0]
                green = fused_image[1]
                red = fused_image[2]  
                # interpolate nir and indexes to match fusioned
                nir = cv2.resize(nir, blue.shape, interpolation=cv2.INTER_CUBIC)
                ndvi = cv2.resize(ndvi, blue.shape, interpolation=cv2.INTER_CUBIC)
                ndwi = cv2.resize(ndwi, blue.shape, interpolation=cv2.INTER_CUBIC)

                print(f'normalizing...')
                nomalized_red = normalize.winsorization_min_max(band=red)
                nomalized_green = normalize.winsorization_min_max(band=green)
                nomalized_blue = normalize.winsorization_min_max(band=blue)
                nomalized_nir = normalize.winsorization_min_max(band=nir)
                nomalized_ndvi = normalize.winsorization_min_max(band=ndvi)
                nomalized_ndwi = normalize.winsorization_min_max(band=ndwi)

                print(f'\n rgb: {blue.shape} \n dnvi: { ndvi.shape}')


                print('stacking...')
                fused_cbers = np.stack([
                    nomalized_red, 
                    nomalized_green, 
                    nomalized_blue, 
                    nomalized_nir, 
                    nomalized_ndvi, 
                    nomalized_ndwi
                    ])

                csv_metadata['PANSHARPENING'] = fusion
                
                fused_meta = cbers_pan_meta
                fused_meta.update({
                    "count": min(fused_cbers.shape),
                    "height": fused_image.shape[1],
                    "width": fused_image.shape[2],
                    "dtype": np.float32,
                    "nodata": np.nan
                })
    
                
                fused_name = f'{source}_{row['id']}_{csv_metadata['PROCESSO'].replace('/', '-')}_{str(csv_metadata['DATA'])}_{fusion}_normalized_{idx}.tiff'
                csv_metadata['NOME'] = fused_name

                # print(csv_metadata)

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

                idx += 1

            print('ALL DONE')

           














    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get sattellite imagery from desired source')

    parser.add_argument('--source', type=str, required=True, help='cbers, google, sentinel2...')
    parser.add_argument('--test', type=bool, required=False, help='test mode')

    args = parser.parse_args()

    main(source=args.source, test=args.test)

    