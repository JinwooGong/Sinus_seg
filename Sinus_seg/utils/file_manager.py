import os
import gc
import pickle
import json
import nrrd
import pydicom
import numpy as np
from glob import glob
from pathlib import Path
from typing import List, Any


def find_seg_files(folder: str) -> List:
    # 경로 객체로 변환
    root_path = Path(folder)
    dcm_folders = set()  # 중복 제거를 위해 set 사용
    # 모든 하위 디렉토리 순회
    for path in root_path.rglob("*.seg.nrrd"):
        # 파일의 절대 경로를 저장
        # dcm_files.append(str(path.absolute()))
        # 파일이 있는 폴더 경로를 저장
        dcm_folders.add(str(path.parent.absolute()))

    return sorted(list(dcm_folders))

def load_scan(path):
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    # def apply_window(slices, arr, window_center=400, window_width=1800, save_path=None):
    #     new_arr = []
        
    #     for dicom, image in zip(slices,arr):
    #         # Apply windowing
    #         window_min = dicom.WindowCenter - dicom.WindowWidth/2
    #         window_max = dicom.WindowCenter + dicom.WindowWidth/2
            
    #         # Apply rescale slope and intercept if present
    #         if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
    #             image = image * dicom.RescaleSlope + dicom.RescaleIntercept
    #         # Clip the image to the window
    #         image = np.clip(image, window_min, window_max)
        
    #         # # Normalize to 0-255
    #         # image = ((image - window_min) / window_width * 255.0)
    #         # image = np.clip(image, 0, 255)
            
    #         new_arr.append(image)
       
    #     return np.array(new_arr).astype(np.float32)
    
    paths = glob(os.path.join(path,'*.dcm'))
    slices = [pydicom.dcmread(s) for s in paths] 

    # sort by major id
    series = [s.SeriesInstanceUID for s in slices]
    major_slices = [s.SeriesInstanceUID == most_common(series) for s in slices]
    slices = np.array(slices)[major_slices]
    
    # sort by instance number
    instance_list = [float(s.InstanceNumber) for s in slices]
    slices = list(np.array(slices)[np.argsort(instance_list)])
    
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)

    arr = np.asarray([d.pixel_array for d in slices]) # z,y,x
    
    # bone windowing
    # arr = apply_window(slices, arr)
    
    # 첫 번째 DICOM 파일
    dicom = slices[0] 
    
    # 픽셀 간격 및 방향 정보 추출
    pixel_spacing = dicom.PixelSpacing
    slice_thickness = dicom.SliceThickness
    spacing = (*pixel_spacing, slice_thickness)
    
    # ImagePosition (Origin) 계산
    pixel_size = arr.shape # z,y,x
    
    z_mm = pixel_size[0] * spacing[0]
    y_mm = pixel_size[1] * spacing[1]
    x_mm = pixel_size[2] * spacing[2]
    
    z_center = (z_mm-1)/2
    y_center = (y_mm-1)/2
    x_center = (x_mm-1)/2
    
    origin = (x_center, y_center, z_center)
    
    properties ={
        'original_shape' : pixel_size,
        'spacing' : spacing,
        'rescale_intercept' : dicom.RescaleIntercept,
        'rescale_slope' : dicom.RescaleSlope,
        'pixel_spacing' : pixel_spacing,
        'slice_thickness' : slice_thickness,
        'origin' : origin,
        'window_center' : dicom.get('WindowCenter', 2000),
        'window_width' : dicom.get('WindowWidth', 4000),
    }
    
    
    del dicom, spacing, origin, pixel_spacing, slice_thickness
    gc.collect()
    
    arr = np.transpose(arr, (2,1,0)).astype(np.float32) # transpose z,y,x to x,y,z
    
    return len(slices), arr, properties

def load_data(folder: str):
    num_slices, data, properites = load_scan(folder)
    dataset = {
        'num_slices': num_slices,
        'data': data,
        'properties': properites
    }
    label = load_nrrd(folder)
    
    return dataset, label

def load_nrrd(folder: str):
    data = None
    for s in os.listdir(folder):
        if s.endswith('.seg.nrrd'):
            nrrd_path = os.path.join(folder, s)
            data, header = nrrd.read(nrrd_path)
            break
    return data

def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def nifti_files(folder: str, join: bool = True, sort: bool = True) -> List[str]:
    return subfiles(folder, join=join, sort=sort, suffix='.nii.gz')


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj: List[Any], file: str, mode: str = 'wb') -> None:
    try:
        with open(file, mode) as f:
            pickle.dump(obj, f)
        print(f"Has been saved to {file}")
    except Exception as e:
        print(f"pickle save error: {str(e)}")


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def pardir(path: str):
    return os.path.join(path, os.pardir)


def split_path(path: str) -> List[str]:
    """
    splits at each separator. This is different from os.path.split which only splits at last separator
    """
    return path.split(os.sep)
        
# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split

# I am tired of confusing those
subfolders = subdirs
save_pickle = write_pickle
write_json = save_json
