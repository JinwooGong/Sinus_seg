from utils import file_manager
from sinus_manager import SinusManager
if __name__=="__main__":
    root_path = r"D:\\2차년도\\골이식재양"
    sinus_manager = SinusManager(root_path)
    dicom_folders = sinus_manager.get_dicom_folders()
    dicom_folder = dicom_folders[1]
    print(dicom_folder)
    dataset, label = file_manager.load_data(dicom_folder)
    num_slices, data, properties = dataset['num_slices'], dataset['data'], dataset['properties']
    print(data.shape)
    if label is not None:
        print(label.shape)