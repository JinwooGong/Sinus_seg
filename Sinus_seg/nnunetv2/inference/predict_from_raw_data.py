import inspect
import multiprocessing
import os
import shutil
import traceback
from asyncio import sleep
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
import torch

from nnunetv2.configuration import default_num_processes
from nnunetv2.dataloading.data_loader import DataLoader
from nnunetv2.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
class PreprocessAdapter(DataLoader):
    def __init__(
        self,
        list_of_lists: List[List[str]],
        list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
        preprocessor: None,
        output_filenames_truncated: List[str],
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: None,
        num_threads_in_multithreaded: int = 1):
        
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    # def generate_train_batch(self):
    #     idx = self.get_indices()[0]
    #     files = self._data[idx][0]
    #     seg_prev_stage = self._data[idx][1]
    #     ofile = self._data[idx][2]
    #     # if we have a segmentation from the previous stage we have to process it together with the images so that we
    #     # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
    #     # preprocessing and then there might be misalignments
    #     data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
    #                                                             self.configuration_manager,
    #                                                             self.dataset_json)
    #     if seg_prev_stage is not None:
    #         seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
    #         data = np.vstack((data, seg_onehot))

    #     if np.prod(data.shape) > (2e9 / 4 * 0.85):
    #         # we need to temporarily save the preprocessed image due to process-process communication restrictions
    #         np.save(ofile + '.npy', data)
    #         data = ofile + '.npy'

    #     return {'data': data, 'data_properites': data_properites, 'ofile': ofile}
    
def load_what_we_need(model_training_output_dir, use_folds, checkpoint_name):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
    plans = load_json(join(model_training_output_dir, 'plans.json'))
    plans_manager = PlansManager(plans)

    if isinstance(use_folds, str):
        use_folds = [use_folds]

    parameters = []
    for i, f in enumerate(use_folds):
        f = int(f) if f != 'all' else f
        checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                map_location=torch.device('cpu'))
        if i == 0:
            trainer_name = checkpoint['trainer_name']
            configuration_name = checkpoint['init_args']['configuration']
            inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])

    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                       num_input_channels, enable_deep_supervision=False)
    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network, trainer_name



def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder: str,
                          model_training_output_dir: str,
                          use_folds: Union[Tuple[int, ...], str] = None,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          checkpoint_name: str = 'checkpoint_final.pth',
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          folder_with_segs_from_prev_stage: str = None,
                          num_parts: int = 1,
                          part_id: int = 0,
                          device: torch.device = torch.device('cuda')):
    print("\n#######################################################################\nPlease cite the following paper "
          "when using nnU-Net:\n"
          "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
          "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
          "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    if device.type == 'cuda':
        device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data).parameters.keys():
        my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    # recursive_fix_for_json_export(my_init_kwargs)
    # maybe_mkdir_p(output_folder)
    # save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

    if use_folds is None:
        use_folds = auto_detect_available_folds(model_training_output_dir, checkpoint_name)

    # load all the stuff we need from the model_training_output_dir
    parameters, configuration_manager, inference_allowed_mirroring_axes, \
    plans_manager, dataset_json, network, trainer_name = \
        load_what_we_need(model_training_output_dir, use_folds, checkpoint_name)

    # check if we need a prediction from the previous stage
    if configuration_manager.previous_stage_name is not None:
        if folder_with_segs_from_prev_stage is None:
            print(f'WARNING: The requested configuration is a cascaded model and requires predctions from the '
                  f'previous stage! folder_with_segs_from_prev_stage was not provided. Trying to run the '
                  f'inference of the previous stage...')
            folder_with_segs_from_prev_stage = join(output_folder,
                                                    f'prediction_{configuration_manager.previous_stage_name}')
            predict_from_raw_data(list_of_lists_or_source_folder,
                                  folder_with_segs_from_prev_stage,
                                  get_output_folder(plans_manager.dataset_name,
                                                    trainer_name,
                                                    plans_manager.plans_name,
                                                    configuration_manager.previous_stage_name),
                                  use_folds, tile_step_size, use_gaussian, use_mirroring, perform_everything_on_gpu,
                                  verbose, False, overwrite, checkpoint_name,
                                  num_processes_preprocessing, num_processes_segmentation_export, None,
                                  num_parts=num_parts, part_id=part_id, device=device)

    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + dataset_json['file_ending']) if
                                 folder_with_segs_from_prev_stage is not None else None for i in caseids]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, seg_from_prev_stage_files, preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)
    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda')
    # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())

    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian(configuration_manager.patch_size)).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    # go go go
    # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    # export_pool = multiprocessing.Pool(num_processes_segmentation_export)
    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed in mta:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))
                while not proceed:
                    sleep(1)
                    proceed = not check_workers_busy(export_pool, r, allowed_num_queued=len(export_pool._pool))

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            if prediction is None:
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
                            else:
                                prediction += predict_sliding_window_return_logits(
                                    network, data, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device)
                            if len(parameters) > 1:
                                prediction /= len(parameters)

                    except RuntimeError:
                        print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                              'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device)
                        else:
                            prediction += predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device)
                        if len(parameters) > 1:
                            prediction /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()

                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...')
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'

                # this needs to go into background processes
                # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
                #                   save_probabilities)
                print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_softmax, ((prediction, properties, configuration_manager, plans_manager,
                                                          dataset_json, ofile, save_probabilities),)
                    )
                )
                print(f'done with {os.path.basename(ofile)}')
        [i.get() for i in r]

    # we need these two if we want to do things with the predictions like for example apply postprocessing
    shutil.copy(join(model_training_output_dir, 'dataset.json'), join(output_folder, 'dataset.json'))
    shutil.copy(join(model_training_output_dir, 'plans.json'), join(output_folder, 'plans.json'))