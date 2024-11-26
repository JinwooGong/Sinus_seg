import os
from utils.file_manager import join

def get_subfolder_paths(root_path):
    subfolder_paths = []
    
    for item in os.listdir(root_path): # root_path의 모든 항목을 순회
        item_path = join(root_path, item) # 모든 항목의 경로
        if os.path.isdir(item_path):
            subfolder_paths.append(item_path)
            
    return subfolder_paths