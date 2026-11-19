import config
from shape import shape_tools
from raster import indexes, pansharpening, normalize, array_tools
from imagery import stac

import os
import gc
import geopandas as gpd
import numpy as np
from pathlib import Path

from PIL import Image
import rasterio
from datetime import date, timedelta, datetime
from scipy import ndimage
import csv
import cv2
import time

def main(fusion:str='gram-schmidt'):

    source = 'cbers'
    shapes_list = os.listdir(config.SHAPE_PATH)

    save_dir = './PA/CBERS'

    normalization_value = None

    for file in shapes_list:

        print(f'\n WORKING IN {str(file)}\n')

        # get geoDATAFRAME
        gdf = shape_tools.gdf(config.SHAPE_PATH + '/' + file)     

        # iterate over geometries
        for idx, row in gdf.iterrows():
            print(f'\n {idx} - id: {row['id']} - processo: {row['processo']} \n')

            if row['processo'] != '855218/1995':
                continue

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

            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)

            # gdf_area:float = data_frame.to_crs("EPSG:10857").area[0] / 10000
            # target_area:float = ( (config.RESOLUTION * 2)  ** 2 ) / 10000

            # print(f'calculated areas: {gdf_area} / {target_area}')

            # if (gdf_area < target_area) or (gdf_area > 209):
            #     continue



            # gets global imagery
            print(f'getting global imagery...')
            global_imagery, _, metadatas = stac.getData(
                    gdf=data_frame   , 
                    shapefile_name="global", 
                    source='cbers', 
                    timeframe=config.DATETIME, 
                    size=int(config.RESOLUTION // 4),
                    max_size=True, 
                    assets=['BAND3', 'BAND2', 'BAND1', 'BAND4'], 
                    verify=False, 
                    dec_read=1, 
                    getAll=False
                    )
            
            print(f'getting PAN')

            target_date = str(metadatas['DATE'])
            csv_metadata['DATA'] = target_date

            global_pan, temp_pan_meta, _ = stac.getData(
                    gdf=data_frame, 
                    shapefile_name="global_pan", 
                    source='cbers', 
                    timeframe=target_date, 
                    size=config.RESOLUTION, 
                    max_size=True,
                    assets=['BAND0'], 
                    verify=False, 
                    dec_read=1, 
                    getAll=False
                    )

            # makes global pan 2D
            global_pan = global_pan[:, :, 0,]

            print(f'pansharpeenning...')
            
            # pansharpening

            if fusion == 'gsa':
                global_fused, _ = pansharpening.gsa(pan=global_pan, ms=global_imagery)
            elif fusion == 'gram-schmidt':
                reference_gs_pan = pansharpening.gram_schmidt(pan=global_pan, ms=global_imagery, params=True)
                global_fused = pansharpening.gram_schmidt(pan=global_pan, ms=global_imagery, params=reference_gs_pan)

            pansharpening.save_gs_params('cbers_gs_params.npz', reference_gs_pan)

            csv_metadata['PANSHARPENING'] = fusion

            if normalization_value is None:
                normalization_value = normalize.winsorization_min_max(band=global_fused, values=True)    

            ndvi = indexes.ndvi(nir=global_fused[:,:, 3], red=global_fused[:,:, 0])
            ndwi = indexes.ndwi(green=global_fused[:,:, 1], nir=global_fused[:,:, 3])


            # normalize all
            print(f'normalizing...')


            np.savez('normalization_bounds.npz', values=normalization_value)

            exit()

            global_fus_norm = normalize.winsorization_min_max(band=global_fused, values=False)
            
            ndvi = (ndvi + 1) / 2
            ndwi = (ndwi + 1) / 2

            indexes_stack = np.stack((ndvi, ndwi))
            indexes_stack = np.transpose(indexes_stack, (1, 2, 0))

            # add ndvi and ndwi to global imagery
            global_fus_norm = np.dstack([global_fus_norm, indexes_stack])         
            
            # save assets meetadata
            cbers_assets = ['RED', 'GREEN', 'BLUE', 'NIR', 'NDVI', 'NDWI']
            csv_metadata['ASSETS'] = cbers_assets

            # save metadata
            g_f_n_meta = temp_pan_meta
            g_f_n_meta.update({
                'driver': 'GTiff',
                'count': min(global_fus_norm.shape),
                'height': global_fus_norm.shape[0],
                'width': global_fus_norm.shape[1]
            })


            print(f'cleaning residual memory...')

            # print(f'size of GLOBAL MS: {global_imagery.nbytes / (1024 ** 2):.2f} MB')
            # print(f'size of GLOBAL PAN: {global_pan.nbytes / (1024 ** 2):.2f} MB')
            # print(f'size of GLOBAL FUSED: {global_fused.nbytes / (1024 ** 2):.2f} MB')

            # exit()

            del global_imagery
            del global_pan    
            del global_fused
            del metadatas
            # del normalized
            del temp_pan_meta
            del ndvi
            del ndwi
            # del indexes_stack
            collected = gc.collect()

            print(f'')




            # global_fus_norm = np.transpose(global_fus_norm, (2, 0, 1))
            # with rasterio.open(save_dir + '/UNCLASSIFIED/' + 'GLOBAL.tiff', "w", **g_f_n_meta) as dest:
            #         dest.write(global_fus_norm)
            
            # print(f'global saved')

            # global_fus_norm = np.transpose(global_fus_norm, (1, 2, 0))

            # exit()
   
            #
            #   OK ATE AQUI
            #

            # divide global

            spatial_resolution = 2

            gdf_area:float = data_frame.to_crs("EPSG:10857").area[0] / 10000
            target_area:float = ( (config.RESOLUTION * spatial_resolution)  ** 2 ) / 10000

            print(f'gdf area: {gdf_area}')
            print(f'target area: {target_area}')

            if gdf_area > target_area:
                print(f' dividing global imagery...')
                tiles = array_tools.divide(
                    image=global_fus_norm,
                    shape_file=data_frame,
                    transform=g_f_n_meta['transform'],
                    crs=g_f_n_meta['crs'],
                    tile_size=config.RESOLUTION,
                    overlap=128,
                    spatial_resolution=spatial_resolution
                    )
            else:
                tiles = [global_fus_norm]
        
            # create directories
    
            #output_dir = Path(config.SOURCES.get(source)[1])
            #output_dir.mkdir(parents=True, exist_ok=True)
            

            total_parts = len(tiles)
    
            for idx, tile_tuple in enumerate(tiles):

                print(f'\n STARTING FOR {idx} OF {total_parts}\n')

                part = f'{idx}-{total_parts}'
                csv_metadata['PARTE'] = part

                if total_parts == 1:
                    tile = tile_tuple
                else:
                    tile, transform = tile_tuple
                    
                    g_f_n_meta.update({
                        "height": tile.shape[0],
                        "width": tile.shape[1],
                        "transform": transform
                    })
                  
                fused_name = f'{source}_{row['id']}_{csv_metadata['PROCESSO'].replace('/', '-')}_{str(csv_metadata['DATA'])}_{fusion}_normalized_{idx}-{total_parts}.tiff'
                csv_metadata['NOME'] = fused_name

                # print(csv_metadata)

                print('\n writing csv file')
                # WRITES DATA TO CSV
                with open(save_dir + '/metadata.csv', mode='a', encoding='utf-8', newline='') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
                    # csv_writer.writeheader()
                    csv_writer.writerow(csv_metadata)

                print(f'\n saving geotiff as {fused_name}\n')
                # SAVE GEOTIFF FILE
                with rasterio.open(save_dir + '/UNCLASSIFIED/' + fused_name, "w", **g_f_n_meta) as dest:
                    dest.write(tile[:,:, 0], 1)
                    dest.set_band_description(1, 'RED')
                    
                    dest.write(tile[:,:, 1], 2)
                    dest.set_band_description(2, 'GREEN')

                    dest.write(tile[:,:, 2], 3)
                    dest.set_band_description(3, 'BLUE')

                    dest.write(tile[:,:, 3], 4)
                    dest.set_band_description(4, 'NIR')

                    dest.write(tile[:,:, 4], 5)
                    dest.set_band_description(5, 'NDVI')

                    dest.write(tile[:,:, 5], 6)
                    dest.set_band_description(6, 'NDWI')
                
                print(f'saved part {part}')
                


                del global_fus_norm
                del tiles
                collected = gc.collect()
                print()

            print('ALL DONE')