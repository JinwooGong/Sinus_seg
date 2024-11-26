import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def RotateImage(img, angle, scale=1):
    if img.ndim > 2:
        height, width, channel = img.shape
    else:
        height, width = img.shape

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    result = cv2.warpAffine(img, matrix, (width, height))

    return result

def rotate_3d(volume, angle, scale=1):
    if len(volume.shape) == 4:
        volume = volume.squeeze(0)
    rotated_volume = []
    for image in volume:
        rotated_volume.append(RotateImage(image, angle, scale))
    rotated_volume = np.asarray(rotated_volume)
    return rotated_volume

def flip_3d(volume, code=1):
    """
    code = 0 : 상하 반전
    code = 1 : 좌우 반전
    """
    flip_volume = []
    for image in volume:
        flip_volume.append(cv2.flip(image, code))
    flip_volume = np.asarray(flip_volume)
    return flip_volume

def dilate_3d_volume(volume, iterations=1):
    """
    3D 볼륨 데이터를 모든 방향으로 팽창시킵니다.
    
    :param volume: 3D numpy array (입력 볼륨)
    :param iterations: 팽창을 반복할 횟수 (기본값 1)
    :return: 팽창된 3D 볼륨
    """
    # 구조 요소 생성 (3x3x3 큐브)
    structure = np.ones((2, 1, 2))
    
    # 팽창 연산 수행
    dilated = ndimage.binary_dilation(volume, structure=structure, iterations=iterations)
    
    return dilated.astype(volume.dtype)

def adjust_ct_image_without_normalization(image, properties, window_center, window_width):

    # 픽셀 값을 HU 단위로 변환
    intercept = properties['rescale_intercept']
    slope = properties['rescale_slope']
    hu_image = image * slope + intercept
    
    # 윈도잉 적용
    low = window_center - window_width // 2
    high = window_center + window_width // 2
    windowed_image = np.clip(hu_image, low, high)
    
    return windowed_image

def smooth_volume_data(volume_data, sigma=1.0, method='gaussian'):
    """
    3차원 볼륨 데이터를 스무딩하는 함수
    
    Parameters:
    -----------
    volume_data : numpy.ndarray
        스무딩할 3차원 볼륨 데이터 (shape: [depth, height, width])
    sigma : float 또는 sequence of float
        가우시안 필터의 표준편차. 각 차원별로 다른 값 설정 가능
    method : str
        스무딩 방법 ('gaussian', 'mean', 'median')
        
    Returns:
    --------
    numpy.ndarray
        스무딩된 3차원 볼륨 데이터
    """
    
    if not isinstance(volume_data, np.ndarray) or volume_data.ndim != 3:
        raise ValueError("입력 데이터는 3차원 numpy 배열이어야 합니다.")
    
    # 입력 데이터 복사
    smoothed_data = volume_data.copy()
    
    if method == 'gaussian':
        # 가우시안 필터 적용
        smoothed_data = gaussian_filter(smoothed_data, sigma=sigma)
        
    elif method == 'mean':
        # 평균값 필터
        kernel_size = int(sigma * 2)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 홀수로 만들기
            
        from scipy.ndimage import uniform_filter
        smoothed_data = uniform_filter(smoothed_data, size=kernel_size)
        
    elif method == 'median':
        # 중앙값 필터
        from scipy.ndimage import median_filter
        kernel_size = int(sigma * 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        smoothed_data = median_filter(smoothed_data, size=kernel_size)
        
    else:
        raise ValueError("지원되지 않는 스무딩 방법입니다. 'gaussian', 'mean', 'median' 중 하나를 선택하세요.")
    
    return np.transpose(smoothed_data, (2,1,0)).astype(np.uint8)