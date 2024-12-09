import os
import sys
import time
import trimesh
import torch
import numpy as np
import time
from scipy import ndimage
from skimage import morphology, measure

from utils import sampling, load_scan, cropping_roi
from utils.file_manager import join
from predictor import predictor


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU")
    return device

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

def post_processing_v2(volume):
    
    # 각 영역에 대해 개별 처리
    result = np.zeros_like(volume) 
    # values = np.unique(volume[1:]) # 클래스가 많을 경우 사용
    
    values = [1]
    for value in values:
        mask = volume == value
        
        # 라벨링
        labeld_mask, num_features = ndimage.label(mask)

        if num_features > 1:  # 적어도 두 개의 연결 영역이 있는 경우
            # 연결 영역의 크기 계산
            sizes = ndimage.sum(mask, labeld_mask, range(1, num_features+1))
            
            # 가장 큰 두 개의 연결 영역 찾기
            largest_two = np.argsort(sizes)[-2:] + 1
            largest_component = np.logical_or(labeld_mask == largest_two[1], labeld_mask == largest_two[0])
            
            # 모폴로지 연산 및 구멍 채우기
            selem = morphology.ball(3)
            cleaned_component = morphology.binary_closing(largest_component, selem)
            cleaned_component = ndimage.binary_fill_holes(cleaned_component)
            
            # 결과에 현재 값 적용
            result[cleaned_component] = value
            
        elif num_features == 1:  # 연결 영역이 하나만 있는 경우
            largest_component = labeld_mask == 1
            
            # 모폴로지 연산 및 구멍 채우기
            selem = morphology.ball(3)
            cleaned_component = morphology.binary_closing(largest_component, selem)
            cleaned_component = ndimage.binary_fill_holes(cleaned_component)
            
            # 결과에 현재 값 적용
            result[cleaned_component] = value

    return result
def rotate_3d_data_y(points, angle_degrees):
    # 라디안으로 변환
    angle_rad = np.radians(angle_degrees)
    
    # Y축 회전 행렬
    rotation_matrix = np.array([
        [np.cos(angle_rad),  0, -np.sin(angle_rad)],
        [0,                  1, 0                 ],
        [np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
     # 회전 적용
    rotated_points = points @ rotation_matrix

    return rotated_points

def get_bbox_center(vertices):
    """바운딩 박스 중심"""
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    return (min_coords + max_coords) / 2

def save_to_numpy(array, output_path):
    np.save(output_path, array)
    print(f"Numpy file has been saved to {output_path}")
    
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    
    return f"{int(minutes):02d}:{int(seconds):02d}"
    

def trans_tuple(t):
    return tuple(t[x] for x in (2, 1, 0))


def get_new_start_end_pos(properties) -> tuple:
    # 랜드 마크 Cropping 도중 x축 회전으로 인해 start, end position 또한 회전 해야함
    # 회전된 된 Start, End position 값을 Size 값을 통해 계산
    # x축으로 회전되었기 x축 변경 X
    original_shape = properties['original_shape']
    before_start_pos = properties['start_pos']
    before_end_pos = properties['end_pos']

    o_z, o_y, o_x = original_shape
    s_z, s_y, s_x = before_start_pos
    e_z, e_y, e_x = before_end_pos
    
    start_z = o_z - e_z
    end_z = e_z - s_z + start_z
    
    start_y = o_y - e_y
    end_y = e_y - s_y + start_y
    
    start_pos = (start_z, start_y, s_x)
    end_pos = (end_z, end_y, e_x)

    return start_pos, end_pos
def process_volume_data(data, properties):
    
    original_shape = properties['original_shape']
    processed_volume = np.zeros(original_shape)
    start_pos, end_pos = get_new_start_end_pos(properties)

    # 크롭된 영역 삽입
    processed_volume[
        start_pos[0]:end_pos[0],
        start_pos[1]:end_pos[1],
        start_pos[2]:end_pos[2]
    ] = data
    
    return processed_volume
    

def save_stl(image_arr, output_path, properties, cropping:bool=False, smoothing:bool=False):
    
    # vertex 좌표 조정 (DICOM 좌표계에 맞춤)
    pixel_spacing = properties['pixel_spacing']
    slice_thickness = properties['slice_thickness']
    
    # 볼륨 데이터 복사 및 전처리
    volume = image_arr.copy()
    volume = np.transpose(volume, (2,1,0))  # x,y,z -> z,y,x
    
    if cropping:
        volume = process_volume_data(volume, properties).astype(np.uint8)
    
    # Binary closing 적용 (threshold가 None일 때만)

    volume = ndimage.binary_closing(volume, structure=np.ones((3,3,3)))

    # Marching cubes 적용
    verts, faces, normals, _ = measure.marching_cubes(volume, 0)
    
    # 스케일링 적용
    verts = verts * [slice_thickness, pixel_spacing[1], pixel_spacing[0]]
    
    # 중심점 이동
    o_z, o_y, o_x = volume.shape
    new_z, new_y, new_x = (o_z*slice_thickness/2, o_y*pixel_spacing[1]/2, o_x*pixel_spacing[0]/2)
    verts -= (new_z, new_y, new_x)
    
    
    # Y축 기준 90도 회전
    verts = rotate_3d_data_y(verts, 90)
    faces = faces[:, ::-1]
    
    # Mesh 생성
    mesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)
    
    # Smoothing 적용 (필요한 경우)
    if smoothing:
        mesh_obj = trimesh.smoothing.filter_taubin(mesh_obj, iterations=300, nu=0.2,lamb=0.53)
    
    # STL 파일로 저장
    mesh_obj.export(output_path)
    print(f"STL file has been saved to {output_path}")
    
def extract_path_prefix(full_path, target_folder='Sinus'):
    """
    Extracts the path up to and including the specified target folder.

    Args:
    full_path (str): The full path to process.
    target_folder (str): The folder name to stop at (inclusive). Default is 'Sinus'.

    Returns:
    str: The extracted path, or the original path if the target folder is not found.
    """
    # Split the path into parts
    parts = full_path.split(os.sep)
    
    # Find the index of the target folder
    try:
        target_index = parts.index(target_folder)
        # Join the parts up to and including the target folder
        return os.sep.join(parts[:target_index + 1])
    except ValueError:
        # If the target folder is not found, return the original path
        print(f"Warning: '{target_folder}' not found in the path. Returning original path.")
        return full_path
    
def main(exe_path, input_path):
    factor = 4
    print("factor:", factor)
    cropping=False
    start_time = time.time()
    input_path = input_path.replace("\\", "/")
    running_path = os.path.abspath(__file__)
    
    root_path = extract_path_prefix(exe_path, target_folder="Sinus")
    
    print(f"\nDicom path: {input_path}")
    print(f"현재 실행 중인 파일의 경로: {running_path}")
    print(f"Sinus 경로: {root_path}")
        
    # Load Data
    num_slice, original_data, data_properties = load_scan.load_scan(input_path)
    print(f"Original DICOM shape: {original_data.shape}")
    
    # save_stl(original_data, join(input_path,'image.stl'), data_properties, threshold=threshold)
    
    thickness = data_properties['slice_thickness']
    total_length = num_slice*thickness
    data = original_data.copy()
    
    load_data_time = time.time()
    print(f"\nData load time: {format_time(load_data_time-start_time)}")
    print(f"Total length: {total_length}")
    if total_length > 130:
        print("\nStitchg data !!")
        cropping = True
    else:
        print("\n1 Tile data !!")
        print("1 tile data can't use cropping.")
        cropping = False
        
    if cropping:
        print("\nUse cropping...")
        
    plans_file = join(root_path,'plans_lowres.json')
    weight_dir = join(root_path,"trained_model", "lowres_checkpoint_final.pth")
    configuration = "3d_lowres"

    if cropping and total_length > 130:
        # stitching data
        print("\n########## Data Cropping ##########")
        print(f"Before cropping data shape: {data.shape}")
        
        model_path = "C:/Dentium/utils/Sinus/trained_model/landmark_9class.pt"
        print(f"Landmark model path: {model_path}")
        data, start, end = cropping_roi.image_cropping(data, model_path)

        data_properties['start_pos'] = start
        data_properties['end_pos'] = end
        
        cropping = True

    else:
        # 1Tile data 
        print("\n########## Data Downsampling ##########")
        print(f"Before downsampling shape: {data.shape}")
        data, data_properties = sampling.downsampling_image(data, data_properties, factor=factor, order=3)
        print(f"After downsampling shape: {data.shape}")
        
    print(f"\nPlans file: {plans_file}")
    print(f"Model path: {weight_dir}")
    
    prediction_image = predictor(data, 1, 3, 
                                 weight_dir, data_properties, plans_file, configuration,
                                 tile_step_size= 1.0, gpu_usage=False
                                 )
        
    # 후처리    
    # if use_post:
    prediction_image[prediction_image == 2] = 1
    prediction_image = post_processing_v2(prediction_image)

    # Dilation
    # prediction_image = dilate_3d_volume(prediction_image)
    
    if not cropping:
        prediction_image = sampling.upsampling(prediction_image, data_properties, factor, 0)
    
    pred_time = time.time()
    print(f"\n예측 생성 시간: {format_time(pred_time-load_data_time)}")
        
    pred_output_path = join(input_path, f"predicted_sinus.stl")
    if not os.path.isfile(pred_output_path):
        save_stl(prediction_image, pred_output_path, data_properties, cropping, smoothing=True)
    
    # if cropping:
    #     pred_output_path = join(input_path, f"predicted_sinus_crop_factor{factor}.stl")
    # else:
    #     pred_output_path = join(input_path, f"predicted_sinus_factor{factor}.stl")
    # save_stl(prediction_image, pred_output_path, data_properties, cropping, smoothing=True)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"STL 생성 시간: {format_time(end_time - pred_time)}")
    print(f"실행 시간: {format_time(execution_time)}")
    
# if __name__ == '__main__':

    # main(
    #     exe_path="C:\\Dentium\\utils\\Sinus\\exe\\Sinus.exe",
    #     # input_path="C:\\Users\\sw2\\source\\repos\\Sinus\\x64\\Release\\dicom_test\\12_standard_Well",
    #     input_path= "D:\\2차년도\\골이식재양_dev\\GWNU 10 cases SE\\4\\before",
    #     cropping=1
    #     )
