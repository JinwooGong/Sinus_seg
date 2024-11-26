import os
from copy import deepcopy
from typing import Union, List, Tuple
import torch
import numpy as np
from nnunetv2.acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from nnunetv2.utilities.file_and_folder_operations import load_json, isfile, save_pickle
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

def apply_inference_nonlin(logits: Union[np.ndarray, torch.Tensor]) -> \
            Union[np.ndarray, torch.Tensor]:
        """
        logits has to have shape (c, x, y(, z)) where c is the number of classes/regions
        """
        is_numpy = isinstance(logits, np.ndarray)

        if is_numpy:
            logits_torch = torch.from_numpy(logits)
        else:
            logits_torch = logits

        probabilities = torch.softmax(logits_torch, 0)

        if is_numpy:
            probabilities = probabilities.numpy()
        return probabilities
    
def revert_cropping(predicted_probabilities: np.ndarray,
                        bbox: List[List[int]],
                        original_shape: Union[List[int], Tuple[int, ...]]):
        """
        ONLY USE THIS WITH PROBABILITIES, DO NOT USE LOGITS AND DO NOT USE FOR SEGMENTATION MAPS!!!

        predicted_probabilities must be (c, x, y(, z))

        Why do we do this here? Well if we pad probabilities we need to make sure that convert_logits_to_segmentation
        correctly returns background in the padded areas. Also we want to ba able to look at the padded probabilities
        and not have strange artifacts.
        Only LabelManager knows how this needs to be done. So let's let him/her do it, ok?
        """
        # revert cropping
        probs_reverted_cropping = np.zeros((predicted_probabilities.shape[0], *original_shape),
                                           dtype=predicted_probabilities.dtype)
        slicer = bounding_box_to_slice(bbox)
        probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_probabilities
        return probs_reverted_cropping
    
def export_prediction_from_softmax(predicted_array_or_file: Union[np.ndarray, str], properties_dict: dict,
                                   configuration_manager: ConfigurationManager,
                                   plans_manager: PlansManager,
                                   dataset_json_dict_or_file: Union[dict, str] = None,
                                   output_file_truncated: str = None,
                                   save_probabilities: bool = False):
    if isinstance(predicted_array_or_file, str):
        tmp = deepcopy(predicted_array_or_file)
        if predicted_array_or_file.endswith('.npy'):
            predicted_array_or_file = np.load(predicted_array_or_file)
        elif predicted_array_or_file.endswith('.npz'):
            predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
        os.remove(tmp)

    predicted_array_or_file = predicted_array_or_file.astype(np.float32)

    # if isinstance(dataset_json_dict_or_file, str):
    #     dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
   
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    # label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    # segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    
    # print(segmentation.shape)
    # print(np.unique(segmentation))
    
    # print(predicted_array_or_file.shape)
    # seg_3 = predicted_array_or_file.argmax(0)
    # print(seg_3.shape)
    # print(np.unique(seg_3))
    # np.save('seg_3.npy', seg_3)
    probabilities = apply_inference_nonlin(predicted_array_or_file)
    # print(probabilities.shape)
    segmentation = probabilities.argmax(0)
    
    # put result in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'], dtype=np.uint8)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    # print(segmentation_reverted_cropping.shape)
    # print(np.unique(segmentation_reverted_cropping))
    return segmentation_reverted_cropping
    # np.save("seg_crop_0000.npy", segmentation_reverted_cropping)
    # save
    # if save_probabilities:
    #     # probabilities are already resampled

    #     # apply nonlinearity
    #     predicted_array_or_file = apply_inference_nonlin(predicted_array_or_file)

    #     # revert cropping
    #     probs_reverted_cropping = revert_cropping(predicted_array_or_file,
    #                                                             properties_dict['bbox_used_for_cropping'],
    #                                                             properties_dict['shape_before_cropping'])
    #     # $revert transpose
    #     probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
    #                                                                        plans_manager.transpose_backward])
    #     print(probs_reverted_cropping.shape)
    #     print(np.unique(probs_reverted_cropping))
        
    #     np.savez_compressed('seg.npz', probabilities=probs_reverted_cropping)
    #     # save_pickle(properties_dict, output_file_truncated + '.pkl')
    #     del probs_reverted_cropping
    # del predicted_array_or_file

    # rw = plans_manager.image_reader_writer_class()
    # rw.write_seg(segmentation_reverted_cropping, output_file_truncated + '.nii.gz',
    #              properties_dict)


def resample_and_save(predicted: Union[str, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], next_configuration: str) -> None:
    # needed for cascade
    if isinstance(predicted, str):
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    predicted = predicted.astype(np.float32)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)

    np.savez_compressed(output_file, seg=segmentation.astype(np.uint8))
