# from ..timesformer.datasets.poseguided import PoseGuided
from timesformer.datasets import *
from torch.utils.data import Dataset, DataLoader
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
from timesformer.models import build_model
import time
import torch

def test_dataloader():

    args = parse_args()
    # if args.num_shards > 1:
    #    args.output_dir = str(args.job_dir)
    cfg = load_config(args)

    
    # Create the dataset
    dataset = Poseguided(cfg, 'train')
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    print("data loader constructed")

    model = build_model(cfg)
    print("model built")
    
    # Check data loading
    for i,  (inputs, labels, video_idx, meta) in enumerate(dataloader):
        print("iteration:", i, "video index:", video_idx)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
        guess = model(inputs)


if __name__ == "__main__":
    # Run the test
    test_dataloader()

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

