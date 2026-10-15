import stac, shape_tools, config, web_tiles
import geopandas as gpd
import os
from pathlib import Path
import argparse

def main(source, test=False):

    if test:
        print(f'\n TEST MODE \n')
        ##################################################################################
        shape = os.listdir(config.SHAPE_PATH)[0]

        gdf = shape_tools.gdf(shape_file=config.SHAPE_PATH + '/' + shape)

        big=0
        small=0
        id_test=114450
        
        for idx, row in gdf.iterrows():
            
            # if row['id']!=id_test:
            #     continue

            print(f'{idx} - id: {row['id']} - processo: {row['processo']}')
            polygon=row['geometry'].envelope

            print()

            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)


            if row['area_ha'] >= 100.0:

                shapes_divided = shape_tools.divide(shape=data_frame,size=1024,crs="EPSG:4326")
                print(f'dividido em: {len(shapes_divided)}')

                i=0
                for shape in shapes_divided:

                    print(f'working in {row['processo']}_part{i}')
                    shape=shape.to_crs(data_frame.crs)
                    print(shape.crs)


                    i += 1

                    print('done')
                    stac.getData(
                        gdf=shape,
                        shapefile_name=f'{row['processo'].replace('/','-')}_part{i}',
                        source='cbers',
                        aux_shape=data_frame,
                        getAll=False)
            break

        #print(f'big: {big}')
        #print(f'small: {small}')
            
        ##############################################################################
        exit()



    if source!='all':
        output_dir = Path(config.SOURCES.get(source)[1])
        output_dir.mkdir(parents=True, exist_ok=True)

    shapes_list = os.listdir(config.SHAPE_PATH)


    for shape in shapes_list:

        gdf = shape_tools.gdf(config.SHAPE_PATH + '/' + shape)     


        for idx, row in gdf.iterrows():
            print(f'\n {idx} - id: {row['id']} - processo: {row['processo']} \n')

            polygon = row['geometry'].envelope

            data_frame = gpd.GeoDataFrame(geometry=[polygon], crs=gdf.crs)

            shapefile_name = f'processo_{row['processo'].replace('/','_')}'
            print(f'name: {shapefile_name}')

            satellites = ['cbers', 'sentinel2']
            xyz = ['google', 'bing']

            if source in satellites:
                stac.getData(
                    gdf=data_frame, 
                    shapefile_name=shapefile_name, 
                    source=source, 
                    getAll=False
                    )
        
            elif source in xyz:
                web_tiles.getData(
                    gdf=data_frame, 
                    source=source, 
                    shapefile_name=shapefile_name,
                )
        
            elif source=='all':
                for s in satellites:

                    output_dir = Path(config.SOURCES.get(s)[1])
                    output_dir.mkdir(parents=True, exist_ok=True)

                    stac.getData(
                        data_frame, 
                        shapefile_name, 
                        source=s, 
                        getAll=False
                        )
                for s in xyz:

                    output_dir = Path(config.SOURCES.get(s)[1])
                    output_dir.mkdir(parents=True, exist_ok=True)

                    webTiles.getData(
                        geometryData=data_frame, 
                        source=s, 
                        shapefile_name=shapefile_name)
        
            else:
                print(f'invalide source')
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get sattellite imagery from desired source')

    parser.add_argument('--source', type=str, required=True, help='cbers, google, sentinel2...')
    parser.add_argument('--test', type=bool, required=False, help='test mode')

    args = parser.parse_args()

    main(source=args.source, test=args.test)

    