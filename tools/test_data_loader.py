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

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    path_to_feature = '/playpen-nas-ssd2/saved_tensors/dance/normal/upenn_0720_Dance_3_7_frame_aligned_videos_gp03/full_features.pt'
    loaded_thing = torch.load(path_to_feature)
    breakpoint()

    model = build_model(cfg)

    # Check data loading
    # try:
    for i,  (inputs, labels, video_idx, meta) in enumerate(dataloader):
        # frames, pose_features, label, index, {} = batch
        for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        
        # breakpoint()

        guess = model(inputs)
        breakpoint()

        
        # print(f"Batch {i}: Shape: {batch.shape}")
        # if i == 2:  # Check first 3 batches
        #     break
    # except Exception as e:
    #     print(f"Error during data loading: {e}")

    # Measure performance
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        pass
    end_time = time.time()
    print(f"Data loading took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Run the test
    test_dataloader()