from pathlib import Path
from einops import rearrange, repeat
import cv2
import torch
import torchvision.transforms as transforms
import os
import numpy as np
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.parser import load_config, parse_args

# export
DEFAULT_MEAN = [0.45, 0.45, 0.45]
DEFAULT_STD = [0.225, 0.225, 0.225]
# convert video path to input tensor for model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DEFAULT_MEAN,DEFAULT_STD),
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

# convert the video path to input for cv2_imshow()
# transform_plot = transforms.Compose([
#     lambda p: cv2.imread(str(p),cv2.IMREAD_COLOR),
#     transforms.ToTensor(),
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     lambda x: rearrange(x*255, 'c h w -> h w c').numpy()
# ])

transform_plot = transforms.Compose([
    lambda x: np.transpose(x, (1, 2, 0)),  # Convert to (h, w, c) and to numpy array
    lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR),  # Convert RGB to BGR for OpenCV
    transforms.ToPILImage(),  # Convert to PIL Image for torchvision transforms
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    lambda x: rearrange(x*255, 'c h w -> h w c').numpy()
])


def get_frames(path_to_video, num_frames=8):
  "return a list of paths to the frames of sampled from the video"
  path_to_frames = list(path_to_video.iterdir())
  path_to_frames.sort(key=lambda f: int(f.with_suffix('').name[-6:]))
  assert num_frames <= len(path_to_frames), "num_frames can't exceed the number of frames extracted from videos"
  if len(path_to_frames) == num_frames:
    return(path_to_frames)
  else:
    video_length = len(path_to_frames)
    seg_size = float(video_length - 1) / num_frames 
    seq = []
    for i in range(num_frames):
      start = int(np.round(seg_size * i))
      end = int(np.round(seg_size * (i + 1)))
      seq.append((start + end) // 2)
      path_to_frames_new = [path_to_frames[p] for p in seq]
    return(path_to_frames_new)


def create_video_input(path_to_video):
  "create the input tensor for TimeSformer model"
  path_to_frames = get_frames(path_to_video)
  frames = [transform(cv2.imread(str(p), cv2.IMREAD_COLOR)) for p in path_to_frames]
  frames = torch.stack(frames, dim=0)
  frames = rearrange(frames, 't c h w -> c t h w')
  frames = frames.unsqueeze(dim=0)
  return(frames)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_masks(masks_in, np_imgs):
  masks = []
  for mask, img in zip(masks_in, np_imgs):
    mask= cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = show_mask_on_image(img, mask)
    masks.append(mask)
  return(masks)

# export
def combine_divided_attention(attn_t, attn_s):
  ## time attention
    # average time attention weights across heads
  attn_t = attn_t.mean(dim = 1)
    # add cls_token to attn_t as an identity matrix since it only attends to itself 
  I = torch.eye(attn_t.size(-1)).unsqueeze(0)
  attn_t = torch.cat([I,attn_t], 0)
    # adding identity matrix to account for skipped connection 
  attn_t = attn_t +  torch.eye(attn_t.size(-1))[None,...]
    # renormalize
  attn_t = attn_t / attn_t.sum(-1)[...,None]

  ## space attention
   # average across heads
  attn_s = attn_s.mean(dim = 1)
   # adding residual and renormalize 
  attn_s = attn_s +  torch.eye(attn_s.size(-1))[None,...]
  attn_s = attn_s / attn_s.sum(-1)[...,None]
  
  ## combine the space and time attention
  # attn_ts = einsum('tpk, ktq -> ptkq', attn_s, attn_t)
  attn_ts = torch.einsum('tpk, ktq -> ptkq', attn_s, attn_t)

  ## average the cls_token attention across the frames
   # splice out the attention for cls_token
  attn_cls = attn_ts[0,:,:,:]
   # average the cls_token attention and repeat across the frames
  attn_cls_a = attn_cls.mean(dim=0)
  attn_cls_a = repeat(attn_cls_a, 'p t -> j p t', j = 8)
   # add it back
  attn_ts = torch.cat([attn_cls_a.unsqueeze(0),attn_ts[1:,:,:,:]],0)
  return(attn_ts)


class DividedAttentionRollout():
  def __init__(self, model, **kwargs):
    self.model = model
    self.hooks = []

  def get_attn_t(self, module, input, output):
    self.time_attentions.append(output.detach().cpu())
  def get_attn_s(self, module, input, output):
    self.space_attentions.append(output.detach().cpu())

  def remove_hooks(self): 
    for h in self.hooks: h.remove()
    
  def __call__(self, input_tensor):
    # test_loader = loader.construct_loader(cfg, "test")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
    #     input_tensor = inputs.to(device)
    #     # TODO: fix this as needed
    #     break
    
    # print("after loop")
  

    self.model.zero_grad()
    self.time_attentions = []
    self.space_attentions = []
    self.attentions = []
    for name, m in self.model.named_modules():
      if 'temporal_attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_t))
      elif 'attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_s))
    preds = self.model(input_tensor)
    for h in self.hooks: h.remove()
    for attn_t,attn_s in zip(self.time_attentions, self.space_attentions):
      self.attentions.append(combine_divided_attention(attn_t,attn_s))
    p,t = self.attentions[0].shape[0], self.attentions[0].shape[1]
    result = torch.eye(p*t)
    for attention in self.attentions:
      attention = rearrange(attention, 'p1 t1 p2 t2 -> (p1 t1) (p2 t2)')
      result = torch.matmul(attention, result)
    mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1 = p, p2=p)
    mask = mask.mean(dim=1)
    mask = mask[0,1:,:]
    width = int(mask.size(0)**0.5)
    mask = rearrange(mask, '(h w) t -> h w t', w = width).numpy()
    mask = mask / np.max(mask)

    try:
      frames = rearrange(input_tensor, 'b c t h w -> b t c h w').squeeze().numpy()
    except:
      frames = rearrange(input_tensor, 'b c t h w -> b t c h w').cpu().squeeze().numpy()
    
    # denormalize the rgb values from [-2,2] -> [0, 255]
    frames = (frames + 2) * (255 / 4)
    frames = np.clip(frames, 0, 255).astype(np.uint8)

    return (mask, frames)



def main():
    args = parse_args()
    cfg = load_config(args)
    print("cfg fetched")

    # TODO: set random seed for reproducable results
    model = build_model(cfg)
    print("model loaded")

    att_roll = DividedAttentionRollout(model)
    test_loader = loader.construct_loader(cfg, "test")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        input_tensor = inputs.to(device)
        masks, frames = att_roll(input_tensor)

        np_imgs = [transform_plot(p) for p in frames]
        masks = create_masks(list(rearrange(masks, 'h w t -> t h w')), np_imgs)
        stacked_images = np.vstack((np.hstack(np_imgs), np.hstack(masks)))
        
        output_dir = '/playpen-nas-ssd3/nofrahm/proficiency/figures/dance_full_test'

        number_files = len(os.listdir(output_dir))
        output_path = os.path.join(output_dir, 'visual_run_' + str(number_files) + '.png')

        cv2.imwrite(output_path, stacked_images)
    


if __name__ == "__main__":
    main()