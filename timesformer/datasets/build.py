# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from fvcore.common.registry import Registry
import torch

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name.capitalize()

    # NOTE: new code added to preload tensors
    # print("name:", name)
    
    # if name == 'Poseguided':
    #     tmp_dataset = DATASET_REGISTRY.get(name)(cfg, split)
    #     tmp_dataset._construct_loader()
    #     tensor_paths = tmp_dataset._path_to_pose_tensors
        
    #     loaded_tensors = {tensor_path: torch.load(tensor_path, map_location='cpu').detach().squeeze() for tensor_path in tensor_paths}
    #     # breakpoint()

    #     for k in loaded_tensors.keys():
    #         loaded_tensors[k].share_memory_()

    #     return DATASET_REGISTRY.get(name)(cfg, split, preloaded_tensors=loaded_tensors)

    return DATASET_REGISTRY.get(name)(cfg, split)
