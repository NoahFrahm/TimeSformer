# from ..timesformer.datasets.poseguided import PoseGuided
from timesformer.datasets import *
from torch.utils.data import DataLoader
from timesformer.utils.parser import load_config, parse_args
from timesformer.models import build_model, fusion_four_modality
import torch
import os
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def test_dataloader():
    args = parse_args()
    cfg = load_config(args)

    # Create the dataset
    # dataset = Poseguided(cfg, 'train')
    dataset = Moe(cfg, 'train')
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    print("data loader constructed")

    # model = build_model(cfg)
    # print("model built")
    
    # Check data loading
    for i,  (inputs, labels, video_idx, meta) in enumerate(dataloader):
        print("iteration:", i, "video index:", video_idx)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)

        # save tensor as image for testing
        save_prefix = '/playpen-nas-ssd3/nofrahm/proficiency/figures/dataload_visualized'
        depth_save_path = os.path.join(save_prefix, 'depth.png')
        flow_save_path = os.path.join(save_prefix, 'flow.png')
        rgb_save_path = os.path.join(save_prefix, 'rgb.png')
        save_paths = [depth_save_path, '', flow_save_path, rgb_save_path]

        for i, save_path in enumerate(save_paths):
            for j in range(8):
                try:
                    selected_frame = inputs[i][0, :, j, :, :]
                    selected_frame = (selected_frame - selected_frame.min()) / (selected_frame.max() - selected_frame.min()) * 255
                    selected_frame = selected_frame.to(torch.uint8) 
                    image = to_pil_image(selected_frame)
                    image.save(f"{save_path.split('.')[0]}{j}.png")
                except:
                    continue
        break
        # guess = model(inputs)

def test_model_arch():
    args = parse_args()
    cfg = load_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fusion_four_modality(cfg)
    model.to(device)

    B, C, T, H, W = 8, 3, 8, 448, 448
    test_input = torch.rand((B, C, T, H, W)).to(device)

    for i in range(100):
        print("iteration:", i)
        output = model(4 * [test_input])
        breakpoint()
        print("iteration:", i, )


if __name__ == "__main__":
    # Run the test
    test = 0
    if test == 0:
        test_dataloader()
    elif test == 1:
        test_model_arch()


# CUDA_VISIBLE_DEVICES=0 python tools/test_data_loader.py --cfg /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/transformer_fusion_bball.yaml \
#     DATA.PATH_TO_DATA_DIR /playpen-nas-ssd2/data_organization/Basketball_no_pose/moe_no_pose \
#     DATA.CAMERA_VIEW int_v_late \
#     SOLVER.MAX_EPOCH 15 \
#     NUM_GPUS 1 \
#     TRAIN.BATCH_SIZE 1 \
#     TEST.BATCH_SIZE 16 \
#     DATA_LOADER.NUM_WORKERS 1 \
#     MODEL.RGB_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_rgb.yaml \
#     MODEL.RGB_CHECKPOINT /playpen-nas-ssd2/asdunnbe/TimeSformer_egoexo/full_reshape/Basketball-reshape/Basketball_rgb_reshape_int_v_late/checkpoints/checkpoint_epoch_00015.pyth \
#     MODEL.DEPTH_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_depth.yaml \
#     MODEL.DEPTH_CHECKPOINT /playpen-nas-ssd2/asdunnbe/TimeSformer_egoexo/full_reshape/Basketball-reshape/Basketball_rgb_reshape_int_v_late/checkpoints/checkpoint_epoch_00015.pyth \
#     MODEL.FLOW_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_flow.yaml \
#     MODEL.FLOW_CHECKPOINT /playpen-nas-ssd2/asdunnbe/TimeSformer_egoexo/full_reshape/Basketball-reshape/Basketball_flow_reshape_int_v_late/checkpoints/checkpoint_epoch_00015.pyth \
#     OUTPUT_DIR /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/fusion-output/Basketball/Basketball_int_v_late_no_pose_fusion                                    
