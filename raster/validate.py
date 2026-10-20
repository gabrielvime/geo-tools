import numpy as np

'''
Validations of data integrity
'''

def cloudFilter(image, cloud_threshold=0.10, shadow_threshold=0.3, contrast_threshold=15.0, blur_threshold=95.0):

    '''
    Calculates cloud and shadow percentages
    Validates if pass by thresholds
    '''
    
    # float to avoid caculus(ingles ruim) error
    img_float = image.astype(float)
    pixels_totais = img_float[0].size
    
    # rgb mean to get bright
    bright = np.mean(img_float, axis=0)

    # calculates brightness        
    px_cloud = np.count_nonzero(bright > 200) 
    px_shadow = np.count_nonzero(bright < 10) 
    
    # percentage
    cloud_pct = (px_cloud / pixels_totais)
    shadow_pct = (px_shadow / pixels_totais)
    
    # cloud and shadow filter
    if cloud_pct > cloud_threshold or shadow_pct > shadow_threshold:        
        return False

    # contrast filter
    contrast = np.std(bright)
    #print(f'contrsta: {contrast}')
    
    if contrast < contrast_threshold:
        return False
    
    # gradient filter
    gy, gx = np.gradient(bright)
    
    # border magnitude
    gnorm = np.sqrt(gx**2 + gy**2)
    
    # border variance
    sharpness = np.var(gnorm)
    #print(f'sharpness: {sharpness}')
    
    if sharpness < blur_threshold:
        return False
    
    # pass
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