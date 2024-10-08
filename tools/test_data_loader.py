# from ..timesformer.datasets.poseguided import PoseGuided
from timesformer.datasets import *
from torch.utils.data import DataLoader
from timesformer.utils.parser import load_config, parse_args
from timesformer.models import build_model, fusion_four_modality
import torch

def test_dataloader():
    args = parse_args()
    cfg = load_config(args)

    # Create the dataset
    # dataset = Poseguided(cfg, 'train')
    dataset = Moe(cfg, 'train')
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    print("data loader constructed")

    model = build_model(cfg)
    print("model built")
    
    # Check data loading
    for i,  (inputs, labels, video_idx, meta) in enumerate(dataloader):
        print("iteration:", i, "video index:", video_idx)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
        breakpoint()
        guess = model(inputs)

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


# CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
#     --cfg /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/basketball/transformer_fusion_binary_no_pose_bball.yaml \
#     DATA.PATH_TO_DATA_DIR /playpen-nas-ssd2/data_organization/Basketball/moe_no_pose             \
#     DATA.CAMERA_VIEW int_v_late          \
#     NUM_GPUS 1  \
#     TRAIN.BATCH_SIZE 8             \
#     MODEL.RGB_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_rgb.yaml       \      
#     MODEL.DEPTH_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_ \
#     model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_depth.yaml         \    
#     MODEL.FLOW_MODEL_CFG /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/configs/MOE/modality_model_configs/Basketball/int_v_late/TimeSformer_divST_16x16_448_flow.yaml      \       
#     OUTPUT_DIR /playpen-nas-ssd3/nofrahm/proficiency/TimeSformer/output/Basketball/Basketball_int_v_late_no_pose_fusion    

# # Old Version
# def test_dataloader():

#     args = parse_args()
#     # if args.num_shards > 1:
#     #    args.output_dir = str(args.job_dir)
#     cfg = load_config(args)

#     # loaded_tensors = []

#     # Create the dataset
#     dataset = Poseguided(cfg, 'train')
#     # dataset = Egoexo(cfg, 'train')

#     # # Create the DataLoader
#     # dataset._construct_loader()
#     # tensor_paths = dataset._path_to_pose_tensors
        
#     # loaded_tensors = {tensor_path: torch.load(tensor_path, map_location='cpu').detach().squeeze() for tensor_path in tensor_paths}
#     # for k in loaded_tensors.keys():
#     #     loaded_tensors[k].share_memory_()

#     # print("shared memory created")

#     # dataset = Poseguided(cfg, 'train', preloaded_tensors=loaded_tensors)
#     dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
#     print("data loader constructed")

#     # path_to_feature = '/playpen-nas-ssd2/saved_tensors/dance/normal/upenn_0720_Dance_3_7_frame_aligned_videos_gp03/full_features.pt'
#     # loaded_thing = torch.load(path_to_feature)
#     model = build_model(cfg)
#     print("model built")
    
#     # Check data loading
#     for i,  (inputs, labels, video_idx, meta) in enumerate(dataloader):
#         print("iteration:", i, "video index:", video_idx)
#         for i in range(len(inputs)):
#             inputs[i] = inputs[i].cuda(non_blocking=True)
        
#         guess = model(inputs)
#         breakpoint()

#     # Measure performance
#     start_time = time.time()
#     for i, batch in enumerate(dataloader):
#         pass
#     end_time = time.time()
#     print(f"Data loading took {end_time - start_time:.2f} seconds")

