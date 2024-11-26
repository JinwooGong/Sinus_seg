import os
import pickle
import json
import nrrd
from pathlib import Path
from typing import List, Any


def find_dcm_folders(folder: str) -> List:
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

def load_nrrd(folder: str):
    for s in os.listdir(folder):
        if s.endswith('.seg.nrrd'):
            nrrd_path = os.path.join(folder, s)
            break
    # NRRD 파일 읽기
    print(nrrd_path)
    data, header = nrrd.read(nrrd_path)
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
