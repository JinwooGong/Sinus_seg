import gc
import os
import numpy as np
import pydicom
from glob import glob
from collections import Counter
from multiprocessing import Pool


def most_common_mp(lst):
    return Counter(lst).most_common(1)[0][0]

def load_dicom(file_path):
    return pydicom.dcmread(file_path)

def process_slice(slice):
    return float(slice.InstanceNumber), slice.SeriesInstanceUID, slice.pixel_array, slice

def multiprocessing_scan(path, cpu_cnt):
    
    path = glob(os.path.join(path,'*.dcm'))
    
    # 병렬로 DICOM 파일 로드
    with Pool(cpu_cnt) as pool:
        slices = pool.map(load_dicom, path)
    
    # 주요 시리즈 ID 찾기
    series = [s.SeriesInstanceUID for s in slices]
    major_series = most_common_mp(series)
    
    # 병렬로 슬라이스 처리
    with Pool(cpu_cnt) as pool:
        processed = pool.map(process_slice, slices)
        
    # 주요 시리즈의 슬라이스만 선택하고 정렬
    major_slices = [(instance_num, pixel_array, slice) 
                    for instance_num, series_id, pixel_array, slice in processed 
                    if series_id == major_series]
    major_slices.sort(key=lambda x: x[0])  # InstanceNumber로 정렬
    
    arr = np.array([slice[1] for slice in major_slices])
    slices = [slice[2] for slice in major_slices]
    
    # 첫 번째 DICOM 파일
    dicom = slices[0]
    
    # 픽셀 간격 및 방향 정보 추출
    pixel_spacing = dicom.PixelSpacing
    slice_thickness = dicom.SliceThickness
    spacing = (*pixel_spacing, slice_thickness)
    
    # ImagePosition (Origin) 계산
    pixel_size = arr.shape  # z,y,x
    
    z_mm = pixel_size[0] * spacing[0]
    y_mm = pixel_size[1] * spacing[1]
    x_mm = pixel_size[2] * spacing[2]
    
    z_center = (z_mm-1)/2
    y_center = (y_mm-1)/2
    x_center = (x_mm-1)/2
    
    origin = (x_center, y_center, z_center)
    
    properties = {
        'original_shape': pixel_size,
        'spacing': spacing,
        'rescale_intercept': dicom.RescaleIntercept,
        'rescale_slope': dicom.RescaleSlope,
        'pixel_spacing': pixel_spacing,
        'slice_thickness': slice_thickness,
        'origin': origin,
        'window_center': dicom.get('WindowCenter', 2000),
        'window_width': dicom.get('WindowWidth', 4000),
    }
    
    del dicom, spacing, origin, pixel_spacing, slice_thickness
    gc.collect()
    
    arr = np.transpose(arr, (2,1,0)).astype(np.float32)  # transpose z,y,x to x,y,z
    return len(slices), arr, properties