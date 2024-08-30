# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import torch
import torch.nn as nn
from timesformer.models import build_model, vanilla_build_model
from timesformer.utils.parser import indirect_load_config, parse_args
from torch.nn import functional as F
from einops import rearrange
from .build import MODEL_REGISTRY
from yacs.config import CfgNode as _CfgNode
import timesformer.utils.checkpoint as cu

# https://huggingface.co/blog/AviSoori1x/seemoe
# https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
# https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py#L647


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Expert(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# NOTE: my attempt at implementation
class ExpertChoice(nn.Module):
    """
    Arguments:
        input_tokens (int): number of tokens experts can select from
        capacity_factor (int): on avg how many experts utilize a token
        num_experts (int): the total number of experts
    Returns:
        ...

    Notes:
        the experts each take k tokens of dimension d
        the router decides which tokens the experts get
        the final output in our case can be
            - final feature vector?
            - final classification?
            - k output features?
            - weighted combo?

        I matrix - n index matrix where I[i, j] specifies j-th selected token of the i-th expert (e x k)
        G matrix - denotes the weight of expert for the selected token (e x k)
        P matrix - refers to a one-hot version of I that will be used to gather tokens for each expert (e x k x n)
    """
    
    def __init__(self, input_tokens=4, token_size=768, capacity_factor=2, num_experts=4, num_classes=4, out_method='weighted'):
        super().__init__()

        self.num_experts = num_experts
        self.out_method = out_method
        self.sequence_length = input_tokens
        self.expert_capacity = int((input_tokens * capacity_factor) / num_experts) # number of tokens each expert can take
        self.expert_dims = self.expert_capacity * token_size

        # Expert embedding matrix (Wg)
        self.expert_embeddings = nn.Embedding(num_embeddings=num_experts, embedding_dim=token_size)
        
        # Experts, TODO: ensure args are properly specified here
        self.experts = nn.ModuleList([
            Expert(self.expert_dims, out_features=self.expert_dims) for _ in range(num_experts)
        ])

        if self.out_method == 'weighted': # MLP that takes in tokens and determines expert weights for weighted sum
            self.sum_weights = nn.Sequential(
                Mlp(in_features=self.sequence_length * token_size, hidden_features=self.sequence_length * token_size, out_features=num_experts),
                nn.Softmax(dim=1)
            )
            self.classification_head = Mlp(in_features=self.expert_dims, hidden_features=self.expert_dims, out_features=num_classes)
        elif self.out_method == 'concatenated': # MLP that processes concatenated input
            self.classification_head = Mlp(in_features=num_experts * self.expert_dims, hidden_features=num_experts * self.expert_dims, out_features=num_classes)

    
    def forward(self, x):
        # x -> (b x (n x d)) batch x tokens x token_dim
        B, N, D = x.shape
        
        # Routing
        S = x @ self.expert_embeddings.weight.t() # Token to expert score (b x n) x e
        S = F.softmax(rearrange(S, 'b n e-> b e n',b=B,n=N,e=self.num_experts), dim=2) # Token to expert score (b x n) x e
        G, I = torch.topk(S, self.expert_capacity, dim=2) # Gating matrix b x e x k
        P = I # P = F.one_hot(I, num_classes=self.sequence_length) # Token gathering matrix (b x e x k x n)

        # Select Tokens        
        x_widened = x.unsqueeze(1).expand(-1, self.num_experts, -1, -1)
        selected_tokens = torch.gather(x_widened, 2, P.unsqueeze(-1).expand(-1, -1, -1, D)) 
        selected_tokens = rearrange(selected_tokens, 'b e k d-> e b (k d)',b=B,e=self.num_experts,k=self.expert_capacity,d=D)

        # Apply experts
        expert_results = [expert(selected_tokens[expert_number]) for expert_number, expert in enumerate(self.experts)] # e x (k  d)
        expert_results = torch.stack(expert_results, dim=1) # b x e x (k  d)

        # Final output creation
        if self.out_method == 'weighted':
            x = rearrange(x, 'b n d -> b (n d)',b=B,n=N,d=D)
            weights = self.sum_weights(x)
            weights_reshaped = weights.unsqueeze(-1)
            weighted_sum = (expert_results * weights_reshaped).sum(dim=1)
            # weighted_sum = torch.einsum('i...,i->...', torch.stack(expert_results, dim=0), weights)
            return self.classification_head(weighted_sum)
        elif self.out_method == 'concatenated':
            # TODO: gotta fix this
            return self.classification_head(torch.cat(expert_results, dim=1))


class MultiModalityTimeS(nn.Module):
    """ mixture of expert across four modalities depth, pose, optical flow, rgb

        Args:
            cfg: contains all cfg specifications for the modality models to be built
            tokens: the number of tokens that will come from each modality model
    """

    def __init__(self, fusion_head, rgb_model, pose_model, depth_model, flow_model):
        super().__init__()
 
        # 4 Encoder models 
        self.depth, self.depth_finetune = depth_model
        self.pose, self.pose_finetune = pose_model
        self.flow, self.flow_finetune = flow_model
        self.rgb, self.rgb_finetune = rgb_model

        # fusion head
        self.fusion_head, self.fusion_finetune = fusion_head

    
    def forward(self, x):
        depth, pose, flow, rgb = x

        # breakpoint()
        if self.depth_finetune:depth_tokens = self.depth(depth, get_feature=True)
        else: 
            with torch.no_grad(): depth_tokens = self.depth(depth, get_feature=True)

        # breakpoint()
        if self.pose_finetune: pose_tokens = self.pose(pose, get_feature=True)
        else:
            with torch.no_grad(): pose_tokens = self.pose(pose, get_feature=True)

        # breakpoint()
        if self.flow_finetune: flow_tokens = self.flow(flow, get_feature=True)
        else:
            with torch.no_grad(): flow_tokens = self.flow(flow, get_feature=True)

        # breakpoint()
        if self.rgb_finetune: rgb_tokens = self.rgb(rgb, get_feature=True)
        else:
            with torch.no_grad(): rgb_tokens = self.rgb(rgb, get_feature=True)

        combined = torch.stack((depth_tokens, pose_tokens, flow_tokens, rgb_tokens), 0) # modality x batch x 768
        modality_tokens = combined.permute(1, 0, 2) # batch x modality x 768
        expert_choice = self.fusion_head(modality_tokens)

        return expert_choice


# This helper lets us load params from cfg
def get_attribute(obj, attr_name):
    try:
        # Try to access the attribute
        att_nestings = attr_name.split('.')
        for att_nesting in att_nestings[:-1]:
            obj = getattr(obj, att_nesting)
        return getattr(obj, att_nestings[-1])
    except AttributeError:
        # Return None if the attribute does not exist
        return None


def get_modality_model_settings(cfg):
    """
    Gets the cfg paths and other necessary information to properly build full cfg files for the 4 modality models
    which are all instances of TimesFormer. 

    Ideally we can get this to work so that swapping models is easy and we can even finetune if needed on specific submodels
    
    """
    number_of_shards, shard_id, rng_seed, opts = get_attribute(cfg, 'NUM_SHARDS'), get_attribute(cfg, 'SHARD_ID'), get_attribute(cfg, 'RNG_SEED'), get_attribute(cfg, 'OPTS')
    output_paths = [cfg.OUTPUT_DIR + '/rgb', cfg.OUTPUT_DIR + '/pose', cfg.OUTPUT_DIR + '/depth', cfg.OUTPUT_DIR + '/flow']
    checkpoint_paths = ['rgb_model_checkpoints', 'pose_model_checkpoints', 'depth_model_checkpoints', 'flow_model_checkpoints']
    cfg_keys = ['rgb_model', 'pose_model', 'depth_model', 'flow_model']
    modality_model_cfgs = [cfg.MODEL.RGB_MODEL_CFG, cfg.MODEL.POSE_MODEL_CFG, cfg.MODEL.DEPTH_MODEL_CFG, cfg.MODEL.FLOW_MODEL_CFG]
    return number_of_shards, shard_id, rng_seed, opts, zip(output_paths, checkpoint_paths, cfg_keys, modality_model_cfgs)


@MODEL_REGISTRY.register()
class moe_fusion(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.pretrained=cfg.PRETRAINED

        self.model = ExpertChoice(input_tokens=cfg.MODEL.INPUT_TOKENS, token_size=cfg.MODEL.TOKEN_SIZE, capacity_factor=2, num_experts=cfg.MODEL.NUM_EXPERTS, num_classes=cfg.MODEL.NUM_CLASSES, out_method=cfg.MODEL.OUT_METHOD)

        if self.pretrained:
            ...

       
    def forward(self, x):
        x = self.model(x)
        return x



# TODO: create config objects here for each modality model to then pass to TimeMOE
# we should have different config files for each modality, and ideally different output dirs as well
# so that we can save checkpoints if we finetune modality models
@MODEL_REGISTRY.register()
class fusion_four_modality(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        # TODO: we need to make sure if we want to resume that we must manually update checkpoint file paths

        # cfg settings for models
        number_of_shards, shard_id, rng_seed, opts, modality_model_settings = get_modality_model_settings(cfg)
        output_path = cfg.OUTPUT_DIR

        # Load modality specific Models
        sub_models={}
        for _, checkpoint_path, sub_model_key, modality_cfg in modality_model_settings:
            cur_cfg = indirect_load_config(cfg_file=modality_cfg,
                                            output_dir=output_path,
                                            num_shards=number_of_shards,
                                            shard_id=shard_id,
                                            rng_seed=rng_seed,
                                            opts=opts,
                                            checkpoint_path=checkpoint_path
                                            )
            cur_sub_model = vanilla_build_model(cur_cfg) #builds the model without any GPU sharding or assignment

            #NOTE: doing checkpoint loading like this may lead to issues with finetuning these parts of the model later on
            # we need to ensure that we properly save checkpoints to ensure that we can finetune, read from cfg to check if we are finetuning
            # since we just use submodel cfgs to load weights and model arch we should move finetune arg out to top level cfg so that save
            # cehckpoint is able to read these args to influence saving procedure, as well as save location
            if get_attribute(cur_cfg, 'TRAIN.CHECKPOINT_FILE_PATH'):
                cu.load_checkpoint(cur_cfg.TRAIN.CHECKPOINT_FILE_PATH, cur_sub_model, data_parallel=False)
            sub_models[sub_model_key] = (cur_sub_model, get_attribute(cfg, 'TRAIN.FINETUNE') is True)

        # Load Fusion method
        with open(cfg.MODEL.FUSION_HEAD_CFG, 'r') as file:
            fusion_head_cfg = _CfgNode()._load_cfg_from_file(file)
        fusion_head = vanilla_build_model(fusion_head_cfg)

        if get_attribute(fusion_head_cfg, 'TRAIN.CHECKPOINT_FILE_PATH'): 
            cu.load_checkpoint(fusion_head_cfg.TRAIN.CHECKPOINT_FILE_PATH, fusion_head, data_parallel=False)
        sub_models['fusion_head'] = (fusion_head, get_attribute(fusion_head_cfg, 'TRAIN.FINETUNE') is True)

        # change this to include fusion head, make cfg global so our custom .train() method can tell which submodels to fix
        self.sub_models = sub_models

        # Construct full model
        self.model = MultiModalityTimeS(**sub_models)
        # for name, module in self.model.named_children(): print(f'Name: {name}, Module: {module}')
        # for name, module in self.model.named_children(): print(f'Name: {name}')
        # for child in model.children(): print(f'Name: {child}')
        # for child in model.children(): print(f'Name: {type(child.rgb)}')
       
    def forward(self, x):
        x = self.model(x)
        return x

    def train(self, mode=True):
        # TODO: custom implementation so we can ensure that submodels remain in eval mode
        super().train(mode)  # This sets the whole model to training mode
        # Keep feature extractor modality models in eval mode, TODO: add logic to not force eval if we desire finetuning
        
        for sub_model_name, (sub_model, fine_tune) in self.sub_models.items():
            if not fine_tune: sub_model.eval()





# # After initial training and before saving:
# for name, param in cur_sub_model.named_parameters():
#     print(f'{name}: {param.data.mean()}')

# print("between")

# # After loading the model
# for name, param in cur_sub_model.named_parameters():
#     print(f'{name}: {param.data.mean()}')



# @MODEL_REGISTRY.register()
# class transformer_fusion(nn.Module):
#     def __init__(self, cfg, **kwargs):
#         super().__init__()
#         self.pretrained=True

#         # TODO: create config objects here for each modality model to then pass to TimeMOE
#         # we should have different config files for each modality, and ideally different output dirs as well
#         # so that we can save checkpoints if we finetune modality models
#         number_of_shards, shard_id, rng_seed, opts, sub_model_settings = get_sub_model_settings(cfg)

#         # Load modality specific Models
#         sub_models={}
#         for output_path, checkpoint_path, sub_model_key, modality_cfg in sub_model_settings:
#             cur_cfg = indirect_load_config(cfg_file=modality_cfg,
#                                             output_dir=output_path,
#                                             num_shards=number_of_shards,
#                                             shard_id=shard_id,
#                                             rng_seed=rng_seed,
#                                             opts=opts,
#                                             checkpoint_path=checkpoint_path
#                                             )
#             cur_sub_model = vanilla_build_model(cur_cfg) #builds the model without any GPU sharding or assignment
#             if get_attribute(cfg, 'TRAIN.FINETUNE'): cur_sub_model.eval()
#             else: cur_sub_model.train()
#             sub_models[sub_model_key] = cur_sub_model

#         # Load Late Fusion method
#         # simple load from weigths and a call to correct vanilla build



        
#         breakpoint()
#         # TODO: ensure that we can freeze seperate components, during training based on cfg
#         # for each component
#         # build model with cfg
#         # save built model for input to TimeMOE
#         # create timeMOE by providing built components
#         # ensure when saving checkpoints each components recieves its own seperate file 
        

#         # build each modality model + moe head
#         # for each modality model -> if pretrained, load pretrained
#         # for moe head -> if pretrained, load pretrained
#         # pass the loaded pretrained/not pretrained models to TimeMOE
#         # ensure checkpointing works correctly
#         self.model = TimeMOE(**cfgs)
       
#         # TODO: make sure that this works for this model?, check each model subpart to load cfg
#         # if self.pretrained:
#         #     load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

#     def forward(self, x):
#         x = self.model(x)
#         return x


# #noisy top-k gating
# class NoisyTopkRouter(nn.Module):
#     def __init__(self, n_embed, num_experts, top_k):
#         super(NoisyTopkRouter, self).__init__()
#         self.top_k = top_k
#         #layer for router logits
#         self.topkroute_linear = nn.Linear(n_embed, num_experts)
#         self.noise_linear =nn.Linear(n_embed, num_experts)

    
#     def forward(self, mh_output):
#         # mh_ouput is the output tensor from multihead self attention block (concatenated tokens from 4 encoders)
#         logits = self.topkroute_linear(mh_output)

#         #Noise logits
#         noise_logits = self.noise_linear(mh_output)

#         #Adding scaled unit gaussian noise to the logits
#         noise = torch.randn_like(logits)*F.softplus(noise_logits)
#         noisy_logits = logits + noise

#         top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
#         zeros = torch.full_like(noisy_logits, float('-inf'))
#         sparse_logits = zeros.scatter(-1, indices, top_k_logits)
#         router_output = F.softmax(sparse_logits, dim=-1)
#         return router_output, indices

# # expert choice
# # https://arxiv.org/pdf/2202.09368 (allows the experts to select tokens rather than tokens select experts, nice cross modality fusion?)
# class ExpertsChooseMaskedRouter(MaskedRouter):
#   """Masked matmul router using experts choose tokens assignment.

#   This router uses the same mechanism as in Mixture-of-Experts with Expert
#   Choice (https://arxiv.org/abs/2202.09368): each expert selects its top
#   expert_capacity tokens. An individual token may be processed by multiple
#   experts or none at all.

#   Note: "experts choose routing" should not be used in decoder blocks because it
#   breaks the autoregressive behavior -- the model will learn to cheat by using
#   future token information to improve current token predictions.
#   """

#   def _compute_routing_instructions(self, router_probs: Array,
#                                     padding_mask: Optional[Array],
#                                     expert_capacity: int) -> RouterMask:
#     """Computes masks for the highest probability token per expert.

#     Args:
#       router_probs: <float32>[num_groups, tokens_per_group, num_experts]
#         probabilities used to determine the routing of tokens to the experts.
#       padding_mask: <float32>[num_groups, tokens_per_group] padding logit mask
#         used to identify padding tokens that should be down-weighted by the
#         router.
#       expert_capacity: Each group will send this many tokens to each expert.

#     Returns:
#         Dispatch and combine arrays for routing with masked matmuls.
#     """
#     tokens_per_group = router_probs.shape[1]

#     if padding_mask is not None:
#       # Because experts choose tokens, we mask probabilities corresponding to
#       # tokens before the top-k operation. Note that, unlike for masked-based
#       # tokens-choose routing, the experts here may still choose to select the
#       # (down-weighted) padding tokens.
#       router_probs *= jnp.expand_dims(padding_mask, axis=-1)

#     # vmap over group dimension.
#     router_probs_t = jax.vmap(lambda m: m.transpose())(router_probs)

#     # Top expert_capacity router probability and corresponding token indices for
#     # each expert. Shapes: [num_groups, num_experts, expert_capacity].
#     expert_gate, expert_index = _top_k(router_probs_t, k=expert_capacity)

#     # Convert to one-hot mask of expert indices for each token in each group.
#     # Shape: [num_groups, num_experts, expert_capacity, tokens_per_group].
#     dispatch_mask = jax.nn.one_hot(
#         expert_index, tokens_per_group, dtype=jnp.int32)

#     # Move axes to conform with shape expected by MoeLayer API.
#     # Shape: [num_groups, tokens_per_group, num_experts, expert_capacity]
#     dispatch_mask = jnp.moveaxis(dispatch_mask, 3, 1)

#     # The combine array will be used for combining expert outputs, scaled by the
#     # router probabilities. Shape: [num_groups, num_experts, tokens_per_group,
#     # expert_capacity].
#     combine_array = jnp.einsum(
#         '...ec,...tec->...tec',
#         expert_gate,
#         dispatch_mask,
#         precision=jax.lax.Precision.DEFAULT)

#     # Return to default dtype now that router computation is complete.
#     combine_array = jax.lax.convert_element_type(combine_array, self.dtype)

#     # Each expert is choosing tokens until it reaches full capacity, so we don't
#     # need an auxiliary loading balancing loss for expert choice routing.
#     auxiliary_loss = 0.0

#     return RouterMask(dispatch_mask, combine_array, auxiliary_loss)


# old implementation with manual GPU loading
# class MultiModalityTimeS(nn.Module):
#     """ mixture of expert across four modalities depth, pose, optical flow, rgb

#         Args:
#             cfg: contains all cfg specifications for the modality models to be built
#             tokens: the number of tokens that will come from each modality model
#     """

#     # def __init__(self, cfg, topk, tokens, num_experts=12, classes=4):
#     def __init__(self, fusion_head, rgb_model, pose_model, depth_model, flow_model):
#         super().__init__()

#         fast_setup = False
#         self.fast = fast_setup

#         if fast_setup:
#             if torch.cuda.is_available():
#                 assert torch.cuda.device_count() == 4
#                 self.device = torch.device("cuda")
#             else:
#                 print("Need 4 GPUs to run model in fast setup")
        
#         # 4 Encoder models TODO: hardcode assignment to 4 GPUs
#         # make custom build_model so it doesnt move to GPU instead allow outer build model to move to GPUs
#         #   - assumes that we can fit all 4 + current on one GPU
        
#         self.depth = build_model(cfg.MODEL.DEPTH_MODEL_CFG, gpu_id=0) if fast_setup else vanilla_build_model(cfg.MODEL.DEPTH_MODEL_CFG)
#         self.depth.eval()
#         self.pose = build_model(cfg.MODEL.POSE_MODEL_CFG, gpu_id=1) if fast_setup else vanilla_build_model(cfg.MODEL.POSE_MODEL_CFG)
#         self.pose.eval()
#         self.flow = build_model(cfg.MODEL.FLOW_MODEL_CFG, gpu_id=2) if fast_setup else vanilla_build_model(cfg.MODEL.FLOW_MODEL_CFG)
#         self.flow.eval()
#         self.rgb = build_model(cfg.MODEL.RGB_MODEL_CFG, gpu_id=3) if fast_setup else vanilla_build_model(cfg.MODEL.RGB_MODEL_CFG)
#         self.rgb.eval()

#         # moe block
#         self.expert_mixture = ExpertChoice() # call build model with cfg here too?
#         if fast_setup: self.expert_mixture.cuda(0)

    
#     def forward(self, x):
#         depth, pose, flow, rgb = x

#         # breakpoint()
#         with torch.no_grad():
#             if self.fast: depth = depth.cuda(device=0)
#             depth_tokens = self.depth(depth, get_feature=True)

#         # breakpoint()
#         with torch.no_grad():
#             if self.fast: pose = pose.cuda(device=1)
#             pose_tokens = self.pose(pose, get_feature=True)

#         # breakpoint()
#         with torch.no_grad():
#             if self.fast: flow = flow.cuda(device=2)
#             flow_tokens = self.flow(flow, get_feature=True)

#         # breakpoint()
#         with torch.no_grad():
#             if self.fast: rgb = rgb.cuda(device=3)
#             rgb_tokens = self.rgb(rgb, get_feature=True)

#         # breakpoint()

#         # move all outputs back to same GPU
#         if self.fast:
#             depth_tokens.cuda(0)
#             pose_tokens.cuda(0)
#             flow_tokens.cuda(0)
#             rgb_tokens.cuda(0)

#         combined = torch.stack((depth_tokens, pose_tokens, flow_tokens, rgb_tokens), 0) # modality x batch x 768
#         modality_tokens = combined.permute(1, 0, 2) # batch x modality x 768
#         # modality_tokens = modality_tokens.flatten(start_dim=1) # batch x (modality * 768)

#         expert_choice = self.expert_mixture(modality_tokens)

#         return expert_choice



        