# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av
import torch

# TODO: create a version of this for feature tensor 
def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        #try:
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        #except:
        #  container = None
        return container
    elif backend == "pt":
        # NOTE: we load onto cpu since it seems that video frame are loaded there as well
        # container = torch.load(path_to_vid).squeeze().cpu()
        # try:

        # attatched_container = torch.load(path_to_vid)
        # dettatched_container = attatched_container.detach()
        # del attatched_container
        # dettatched_container.cpu()
        # dettatched_container.requires_grad_(False)
        # container = dettatched_container.squeeze()

        # except:
        #     container = torch.load(path_to_vid)

        container = torch.load(path_to_vid, map_location='cpu').detach()
        container.requires_grad_(False)
        container = container.squeeze()

        # container = torch.zeros(2000, 17, 128)
        
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))
