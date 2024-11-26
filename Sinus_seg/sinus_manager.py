from pathlib import Path
from typing import List

def get_data(folder: str) -> List:
    # 경로 객체로 변환
    root_path = Path(folder)
    dcm_folders = set()  # 중복 제거를 위해 set 사용
    
    # 모든 하위 디렉토리 순회
    for path in root_path.rglob("*.dcm"):
        # 파일의 절대 경로를 저장
        # dcm_files.append(str(path.absolute()))
        # 파일이 있는 폴더 경로를 저장
        dcm_folders.add(str(path.parent.absolute()))
        

    return sorted(list(dcm_folders))

class SinusManager:
    def __init__(self, root_path):
        self.dicom_folders = get_data(root_path)
        
    def get_dicom_folders(self):
        return self.dicom_folders
