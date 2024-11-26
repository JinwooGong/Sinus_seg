import gc
import torch
import warnings
import numpy as np
from scipy import ndimage
from skimage import morphology

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, predict_sliding_window_return_logits
from nnunetv2.inference.export_prediction import export_prediction_from_softmax
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

def load_what_we_need(plans_manager:PlansManager,checkpoint_path, num_input_channels=1, num_output_channels=3, device = torch.device('cuda')):
    # dataset_json = load_json('dataset.json')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
        'inference_allowed_mirroring_axes' in checkpoint.keys() else None
    parameters = checkpoint['network_weights']
    configuration_manager = plans_manager.get_configuration(configuration_name)
    
    network = get_network_from_plans(configuration_manager, num_input_channels, num_output_channels, deep_supervision=True)
    
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, network, trainer_name

def predict_from_raw_data_v2(data,
                             in_channels: int,
                             out_channels: int,
                          plans_manager: PlansManager,#   model_training_output_dir: str,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          checkpoint_path: str = 'trained_model/checkpoint_final.pth',
                          num_seg_heads: int = 3,
                          device: torch.device = torch.device(type='cuda')
                        ):
    if use_gaussian:
        print("Using gaussian filter")
        
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, network, trainer_name = \
        load_what_we_need(plans_manager, checkpoint_path, num_input_channels=in_channels, num_output_channels=out_channels, device=device)
    # precompute gaussian    
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    
    if device.type == 'cpu':
        perform_everything_on_gpu = False
    if perform_everything_on_gpu:
        device = torch.device(type='cuda')  # set the desired GPU with CUDA_VISIBLE_DEVICES!
        inference_gaussian = inference_gaussian.to(device)   

    if device.type=='cuda':
        network.to(device)
    network.load_state_dict(parameters)
    if not isinstance(data, torch.Tensor):
                # pytorch will warn about the numpy array not being writable. This doesnt matter though because we
                # just want to read it. Suppress the warning in order to not confuse users...
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)

    prediction = predict_sliding_window_return_logits(
        network, data, num_seg_heads,
        configuration_manager.patch_size,
        mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        precomputed_gaussian=inference_gaussian,
        perform_everything_on_gpu=perform_everything_on_gpu,
        verbose=verbose,
        device=device)
    del network
    gc.collect()

    return prediction
    
def _normalize(data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager, foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
 
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = ZScoreNormalization
            if normalizer_class is None:
                raise RuntimeError('Unable to locate class \'%s\' for normalization' % scheme)
        
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data
    
    
def run_case(data, seg, data_properties, plans_manager, configuration_manager, verbose=1):
    
    data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    original_spacing = [data_properties['spacing'][i] for i in plans_manager.transpose_forward]
    # crop, remember to store size before cropping!
    shape_before_cropping = data.shape[1:]
    data_properties['shape_before_cropping'] = shape_before_cropping
    
    # this command will generate a segmentation. This is important because of the nonzero mask which we may need
    data, seg, bbox = crop_to_nonzero(data, seg)
    data_properties['bbox_used_for_cropping'] = bbox
    
    data_properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

    # resample
    target_spacing = configuration_manager.spacing  # this should already be transposed

    if len(target_spacing) < len(data.shape[1:]):
        # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
        # in 3d we do not change the spacing between slices
        target_spacing = [original_spacing[0]] + target_spacing
    new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    data = _normalize(data, seg, configuration_manager, plans_manager.foreground_intensity_properties_per_channel)
   
    old_shape = data.shape[1:]
    data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)

    if verbose:
        print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
            f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

    if np.max(seg) > 127:
        seg = seg.astype(np.int16)
    else:
        seg = seg.astype(np.int8)
    return data, seg, data_properties




def predictor(
    data:np.array, in_channels:int, out_channels:int,
    weight_dir:str, data_properties:dict, plans_file:str, configuration:str,
    tile_step_size: int = 1.0, gpu_usage: bool = False,
    ):
    """
    data format: (b, x, y, n)
    """
    plans_manager = PlansManager(plans_file) 
    configuration_manager = plans_manager.get_configuration(configuration)
    
    if len(data.shape) == 3:
        data = data[None]
        
    # 데이터 전처리
    preprocessed_data, _, data_properties = run_case(data, None, data_properties, plans_manager, configuration_manager)
    del data # 원본 data 삭제
    gc.collect()
    
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    device = torch.device("cpu")

    pred_image = predict_from_raw_data_v2(preprocessed_data,
                            in_channels,
                            out_channels,
                            plans_manager, # '/data1/jwkong/sinus/nnUNet_results/Dataset001_UserTest/nnUNetTrainer__nnUNetPlans__3d_fullres',
                            tile_step_size=tile_step_size,
                            use_gaussian=True,
                            use_mirroring=False,
                            perform_everything_on_gpu=gpu_usage,
                            verbose=True,
                            # checkpoint_name='checkpoint_final.pth',
                            checkpoint_path=weight_dir,
                            device=device)
    del preprocessed_data
    gc.collect()
    
    pred_image = np.array(pred_image)

    prediction_image = export_prediction_from_softmax(pred_image, data_properties,
                                configuration_manager,
                                plans_manager,
                                )
    
    # # 후처리
    # prediction_image = prediction_image.astype(np.int8)
    # prediction_image[prediction_image == 2] = 1
    # prediction_image = post_processing_v2(prediction_image)

    # Dilation
    # if dilation:
    #     print("Using Dilation...")
    #     prediction_image = transform.dilate_3d_volume(prediction_image)
    
    print(f"Prediction Image Shape: {prediction_image.shape}")
    
    return prediction_image.astype(np.uint8)