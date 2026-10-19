"""
Configurations file
"""

from pystac_client import Client

SOURCES = {
    'cbers': ['CBERS-4A FUSED', 'cbers_imagery'], 
    'sentinel2': ['Sentinel-2 L2A (AWS)', 'sentinel2_imagery'],
    'google': ['Google XYZ Tiles', 'google_imagery-grids'],
    'bing': ['Bing STAC', 'bing_imagery']}
"""'source':['Source Name','source_path']"""

### WINDOW CONFIGURATIONS ###
SQUARE=True
RESOLUTION = 640            # final square resolution
EXPAND_FACTOR = 1.0         # expansion factor (1.0 means original size), not recommended to mess with this
ZOOM = 18                   #XYZ Tiles Zoom
MAX_SIZE=False

DATETIME='2023-02-03/2026-08-30'
SHAPE_PATH = 'shapes'

### POLYGON DRAW CONFIGURATIONS ##
DRAW_POLYGON= False
POLYGON_COLOR = 'red'
POLYGON_WIDTH = 5


##### CBERS CONFIGURATIONS #####
def cbers(bbox):

    catalog = Client.open("https://data.inpe.br/bdc/stac/v1")

    search = catalog.search(
        collections=["CB4A-WPM-PCA-FUSED-1"],   # collection
        bbox=bbox,                              # bounding box
        datetime=DATETIME)                      #time frame

    return list(search.items())


##### SENTINEL-2 CONFIGURATIONS #####
def sentinel2(bbox):
    catalog = Client.open("https://earth-search.aws.element84.com/v1")  

    search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=DATETIME,
    query={"eo:cloud_cover": {"lt": 10}}  # cloud filter
    )

    return list(search.items())



