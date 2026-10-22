import numpy as np

'''
Validations of data integrity
'''

def cloud_filter(image, cloud_threshold=0.10, shadow_threshold=0.30):

    '''
    Calculates cloud and shadow percentages
    Validates if pass by thresholds
    '''
    
    # float to avoid caculus(ingles ruim) error
    img_float = image.astype(float)
    total_pixel = img_float[0].size
    
    # rgb mean to get bright
    bright = np.mean(img_float, axis=0)

    # calculates brightness        
    px_cloud = np.count_nonzero(bright > 200) 
    px_shadow = np.count_nonzero(bright < 10) 
    
    # percentage
    cloud_pct = (px_cloud / total_pixel)
    shadow_pct = (px_shadow / total_pixel)
    
    # cloud and shadow filter
    if cloud_pct > cloud_threshold or shadow_pct > shadow_threshold:
        print(f'cloud: {cloud_pct} / shadow: {shadow_pct}')        
        return False
    else:
        return True

   
def contrast_filter(image, threshold=0.1):
    
    # float to avoid caculus(ingles ruim) error
    img_float = image.astype(float)
    
    # rgb mean to get bright
    bright = np.mean(img_float, axis=0)

    # contrast filter
    contrast = np.std(bright)/100
    #print(f'contrsta: {contrast}')
    
    if contrast < threshold:
        print(f'contrast: {contrast}')
        return False
    else:
        return True


def sharpness_filter(image, threshold=0.1):

    # float to avoid caculus(ingles ruim) error
    img_float = image.astype(float)
    
    # rgb mean to get bright
    bright = np.mean(img_float, axis=0)

    # gradient filter
    gy, gx = np.gradient(bright)
    
    # border magnitude
    gnorm = np.sqrt(gx**2 + gy**2)
    
    # border variance
    sharpness = np.var(gnorm)/100
    #print(f'sharpness: {sharpness}')
    
    if sharpness < threshold:
        print(f'sharpness: {sharpness}')
        return False
    
    else:
        return True


def dataIntegrity(image, threshold=0.10):
    '''
    Verify data integrity
    Images with no data
    '''

    total_pixels = image[0].size
    empty_pixels = np.count_nonzero(image[0] == 0)
    
    if (empty_pixels / total_pixels) > 0.10:
        return False
    else:
        return True