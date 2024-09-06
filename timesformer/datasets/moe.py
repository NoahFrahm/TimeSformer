# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Moe(torch.utils.data.Dataset):
    # TODO: fix to be accurate to how we load data from this dataset, update doc string etc.
    """
    Mixture of experts video loader. Construct the MOE video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """


    def __init__(self, cfg, mode, num_retries=10, preloaded_tensors={}):
        """
        Construct the PoseGuided video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_pose_video_1 path_to_depth_video_1 path_to_flow_video_1 path_to_rgb_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for PoseGuided".format(mode)
        self.mode = mode
        self.cfg = cfg

        self.pose_tensors = preloaded_tensors

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing MOE {}...".format(mode))
        self._construct_loader()


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.CAMERA_VIEW, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_rgb_videos = []
        self._path_to_pose_videos = []
        self._path_to_depth_videos = []
        self._path_to_flow_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    # NOTE: 4 videos + 1 label
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 5
                )
                pose_video_path, depth_video_path, flow_video_path, rgb_video_path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
        
                # TODO: update this to make different places to save modality video paths
                for idx in range(self._num_clips):
                    self._path_to_rgb_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, rgb_video_path)
                    )
                    self._path_to_pose_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, pose_video_path)
                    )
                    self._path_to_depth_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, depth_video_path)
                    )
                    self._path_to_flow_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, flow_video_path)
                    )

                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._labels) > 0
        ), "Failed to load MOE split {} from {}".format(
            self._split_idx, path_to_file
        )

        logger.info(
            "Constructing MOE dataloader (size: {}) from {}".format(
                len(self._labels), path_to_file
            )
        )


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1 # TODO: figure out how we can use this to extract correct pose feature
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            rgb_video_container, pose_video_container, flow_video_container, depth_video_container = None, None, None, None
            
            try: # rgb video portion
                rgb_video_container = container.get_video_container(
                    self._path_to_rgb_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load rgb video from {} with error {}".format(
                        self._path_to_rgb_videos[index], e
                    )
                )
            
            try: # pose video portion
                pose_video_container = container.get_video_container(
                    self._path_to_pose_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load pose video from {} with error {}".format(
                        self._path_to_pose_videos[index], e
                    )
                )
            
            try: # flow video portion
                flow_video_container = container.get_video_container(
                    self._path_to_flow_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load flow video from {} with error {}".format(
                        self._path_to_flow_videos[index], e
                    )
                )

            try: # depth video portion
                depth_video_container = container.get_video_container(
                    self._path_to_depth_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load depth video from {} with error {}".format(
                        self._path_to_depth_videos[index], e
                    )
                )
            
            # Select a random video if the current video was not able to access.
            if rgb_video_container is None or pose_video_container is None or depth_video_container is None or flow_video_container is None:
                
                if rgb_video_container is None:
                    logger.warning(
                        "Failed to load rgb video idx {} from {}; trial {}".format(
                            index, self._path_to_rgb_videos[index], i_try
                        )
                    )

                if pose_video_container is None:
                    logger.warning(
                        "Failed to load pose video idx {} from {}; trial {}".format(
                            index, self._path_to_pose_videos[index], i_try
                        )
                    )

                if depth_video_container is None:
                    logger.warning(
                        "Failed to load depth video idx {} from {}; trial {}".format(
                            index, self._path_to_depth_videos[index], i_try
                        )
                    )

                if flow_video_container is None:
                    logger.warning(
                        "Failed to load flow video idx {} from {}; trial {}".format(
                            index, self._path_to_flow_videos[index], i_try
                        )
                    )

                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            # TODO: the video container is a subclip, make sure we extract the pose feature in the same way with the same exact params?
            # aka we need one to one correspondance between frame in video and pose feature
            # pose tokens TODO: create decode call here that can get corresponding pose features
            # Decode video. Meta info is used to perform selective decoding.
            modality_frames = decoder.multi_video_decode( 
                [rgb_video_container, pose_video_container, depth_video_container, flow_video_container],
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            
            modified_frames = []
            for frames in modality_frames:
                if frames is None:
                    logger.warning(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                label = self._labels[index]

                # Perform color normalization.
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

                # NOTE: we will be using TimesFormer modality models so no need for this
                # if not self.cfg.MODEL.ARCH in ['vit']:
                #     frames = utils.pack_pathway_output(self.cfg, frames)
                # else:

                # Perform temporal sampling from the fast pathway.
                frames = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                    ).long(),
                )
                modified_frames.append(frames)
            return modified_frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )


    def __len__(self):
        """
        Returns:
            (int): the number of data points in the dataset.
        """
        return len(self._labels)
