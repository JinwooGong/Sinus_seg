import torch
import numpy as np
import pandas as pd
from skimage.transform import resize

from utils import load_scan

landmark_dict = { # 9 class
    0 : "Na", # 0
    1 : "ANS", # 1
    19 : "Po_R", # 2
    20 : "Po_L", # 3
    21 : "Or_R", # 4
    22 : "Or_L", # 5
    31 : "Mid_1", # 6
    32 : "PNS", # 7 Mid_2에서 변경
    35 : "If", # 8
}

# landmark_dict = { # 11 class
#     1 : "ANS", # 0
#     19 : "Po_R", # 1
#     20 : "Po_L", # 2
#     21 : "Or_R", # 3
#     22 : "Or_L", # 4
#     31 : "Mid_1", # 5
#     32 : "Mid_2", # 6
#     45 : "Ala_R",
#     46 : "Ala_L",
#     47 : "Tragus_R",
#     48 : "Tragus_L"
# }

# landmark_dict = { # 17 class
#     0 : "Na", # 0
#     1 : "ANS", # 1
#     19 : "Po_R", # 2
#     20 : "Po_L", # 3
#     21 : "Or_R", # 4
#     22 : "Or_L", # 5
#     23 : "Max.1_R", # 6
#     24 : "Max.1_L", # 7
#     25 : "Max.6_R", # 8
#     26 : "Max.6_L", # 9
#     31 : "Mid_1", # 10
#     32 : "PNS", # 11 PNS로 명칭 변경 (이전 : Mid_2)
#     35 : "If", # 12
#     45 : "Ala_R", # 13
#     46 : "Ala_L", # 14
#     47 : "Tragus_R", # 15
#     48 : "Tragus_L" # 16
# }

def landmark_detection(data, model_path):

    image_scale = (256, 256, 256)
    model = torch.load(model_path,map_location=torch.device('cpu'))
    size = image_scale
    
    # z축 기준으로 90도 회전
    dcm_data = np.rot90(data, k=-1, axes=(1, 2))
    dcm_data = dcm_data.transpose(1,2,0)
    dcm_data = dcm_data[:,:,::-1] # z출 Flip
    
    x,y,z = dcm_data.shape

    hu_min = np.min(dcm_data)
    hu_max = np.max(dcm_data)
    
    # 데이터 정규화
    dcm_data = (dcm_data - hu_min)/(hu_max - hu_min) # overflow
    dcm_res = resize(dcm_data,(256,256,256), anti_aliasing=True)
    dcm_res = dcm_res * (hu_max - hu_min) + hu_min # 리사이즈된 데이터를 원래의 HU 값 범위로 복원

    img =dcm_res.astype(np.float32)
    img1 = (img - np.min(img))/(np.max(img) - np.min(img)) # 이미지를 다시 정규화
    img1 = (img1 - np.mean(img1)) / np.std(img1) # 이미지 표준화 (평균 0, 표준편차 1)

    out = model(torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float())

    idx_pred = []
    for k in range(len(out[0])):
        # 모델 출력에서 각 랜드마크의 위치를 찾고, 원래 이미지 크기에 맞게 스케일을 조정한다.
        r,c,h = np.unravel_index(np.argmax(out[0][k].detach().cpu().numpy()), out[0][k].shape) # 모델에서 아마 x y z 좌표
        idx_pred.append([h/64*x,c/64*y,r/64*z]) # 원래 스케일로 변경 256

    lab_pred = np.asarray(idx_pred)

    return lab_pred, data # z,y,x

def crop_3d_image(image, start_coords, end_coords):
    """
    Crop a 3D image based on start and end coordinates.
    
    Parameters:
    image (numpy.ndarray): 3D image array
    start_coords (tuple): Starting coordinates (x, y, z)
    end_coords (tuple): Ending coordinates (x, y, z)
    
    Returns:
    numpy.ndarray: Cropped 3D image
    """
    z1, y1, x1 = start_coords
    z2, y2, x2 = end_coords
    
    return image[z1:z2, y1:y2, x1:x2]

def image_cropping(data, model_path):
    data = np.transpose(data, (2,1,0)) # x,y,z->z,y,x
    
    lab_pred, data = landmark_detection(data, model_path)
    
    landmark_list = [value for key, value in landmark_dict.items()]

    lm_dict = {lm : pred for lm, pred in zip(landmark_list, lab_pred)}
    
    points = (
        lm_dict['ANS'][[2,1,0]].astype(np.int32),
        lm_dict['Po_R'][[2,1,0]].astype(np.int32),
        lm_dict['Po_L'][[2,1,0]].astype(np.int32),
    )
    
    dcm_data = np.flip(data, axis=0) # z축 flip
    dcm_data = np.flip(dcm_data, axis=1) # y축 flip
    
    ans_z, ans_y, ans_x = points[0]
    pr_z, pr_y, pr_x = points[1]
    pl_z, pl_y, pl_x = points[2]

    gap = 40
    back = min(pr_y, pl_y) + gap
    top = min(pr_z, pl_z)
    front = ans_y + gap
    right = pr_x
    left = pl_x
    bottom = 100
    
    if abs(top-ans_z) <=50:
        top += 50
    
    right, left = (right , left) if left > right else (left, right)
    start = (bottom, back, right) # z1, y1, x1
    end = (top, front, left) # z2, y2, x2

    print(f"Start Position (z,y,x): {start}, End Position (z,y,x): {end}")
    cropped_image = crop_3d_image(dcm_data, start, end)
    tr_cropped_image = np.flip(cropped_image, axis=1)
    tr_cropped_image = np.flip(tr_cropped_image, axis=0)
    
    tr_cropped_image = np.transpose(tr_cropped_image, (2,1,0)) # transpose z, y, x to x,y,z
    print(f"cropped image shape (x,y,z): {tr_cropped_image.shape}")
    return tr_cropped_image, start, end

def label_cropping(label, start, end):
    """
    input shape: (x, y, z)
    output shape: (x, y, z)
    """
    
    label_data = np.transpose(label, (2,1,0)) # transpose x, y, z to z, y ,x
    label_data = np.flip(label_data, axis=0)
    label_data = np.flip(label_data, axis=1)
    
    cropped_label = crop_3d_image(label_data, start, end)
    
    tr_cropped_label = np.flip(cropped_label, axis=1)
    tr_cropped_label = np.flip(tr_cropped_label, axis=0)
    
    tr_cropped_label = np.transpose(tr_cropped_label, (2,1,0)) # trnaspose z, y, x to x, y, z
    print(f"cropped label shape (x,y,z): {tr_cropped_label.shape}")
    return tr_cropped_label