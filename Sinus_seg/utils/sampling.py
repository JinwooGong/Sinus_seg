import numpy as np
from scipy.ndimage import zoom
import gc

def downsampling_image(image, properties, factor =2, order=3, is_label=False):
    image = np.transpose(image, (2,1,0))
    image_shape = image.shape
    original_shape = np.array(image_shape)
    original_spacing = np.array(properties['spacing'])
    

    # Calculate new shape (1/4 of original in each dimension)
    new_shape = np.ceil(original_shape / factor).astype(int)

    # Calculate new spacing
    new_spacing = original_spacing[1] * (original_shape[1] / new_shape[1])
    
    target_shape = tuple(new_shape)
    new_spacing = tuple([new_spacing]*3)

    if not is_label:
        properties['original_shape'] = image_shape
        properties['original_spacing'] = original_spacing
        properties['target_shape'] = target_shape
        properties['spacing'] = new_spacing
        
    resize_factors = [t / c for t, c in zip(target_shape, original_shape)]

    downsampled_image = zoom(image, resize_factors, order=order)
    
    del image, resize_factors, target_shape, new_spacing, original_shape, original_spacing
    gc.collect()
    
    downsampled_image = np.transpose(downsampled_image, (2,1,0))
    return downsampled_image, properties

def upsampling(image, properties, factor=2, order=1):
    image = np.transpose(image, (2,1,0))
    # 레이블의 경우 properties에서 직접 원본 크기 정보를 사용
    original_shape = properties['original_shape']
    current_shape = image.shape
    resize_factors = [o / c for o, c in zip(original_shape, current_shape)]
    

    # scipy.ndimage.zoom을 사용하여 업샘플링
    upsampled_image = zoom(image, resize_factors, order=order)
    
    # 메모리 정리
    del image, resize_factors
    gc.collect()
    
    # (x,y,z) 순서로 다시 변경
    upsampled_image = np.transpose(upsampled_image, (2,1,0))

    return upsampled_image
    