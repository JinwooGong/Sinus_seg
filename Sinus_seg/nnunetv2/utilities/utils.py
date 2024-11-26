from typing import Union
import os
from nnunetv2.utilities.file_and_folder_operations import *
import numpy as np
import re
import time

def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files must be .nii.gz and have 4 digit channel index
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(f + "_\d\d\d\d" + file_ending)
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists


def extract_3d_region(data, center_x, center_y, center_z, spacing=None):
    
    # 이미지 크기 확인
    depth, height, width = data.shape
    
    # 512x512x512 기준으로 비율 계산
    ratio = 120 / 512
    
    # 실제 추출할 영역 계산
    offset_x = int(ratio * width)
    offset_y = int(ratio * height)
    offset_z = int(ratio * depth)
    
    # 추출할 영역의 좌표 계산
    x_start = max(0, center_x - offset_x)
    x_end = min(width, center_x + offset_x)
    y_start = max(0, center_y - offset_y)
    y_end = min(height, center_y + offset_y)
    z_start = max(0, center_z - offset_z)
    z_end = min(depth, center_z + offset_z)
    
    # 영역 추출
    extracted_region = data[z_start:z_end, y_start:y_end, x_start:x_end]
    print(f"사용된 비율: x={offset_x/width:.4f}, y={offset_y/height:.4f}, z={offset_z/depth:.4f}")

    return extracted_region

def cropping_1tile(data, x_point = 0, y_point = -70, z_point = 80):
    start_time = time.time()
    data_shape = data.shape
    # 크로핑 중심점 계산
    center_z = data_shape[0]//2 + z_point  # 중심점 z 좌표
    center_y = data_shape[1]//2 + y_point  # 중심점 y 좌표
    center_x = data_shape[2]//2 + x_point # 중심점 x 좌표

    extracted_image = extract_3d_region(data, center_x, center_y, center_z)

    extracted_image = extracted_image.astype(np.float32)

    end_time = time.time()
    
    print(f"Time: {end_time - start_time}")
    
    return extracted_image