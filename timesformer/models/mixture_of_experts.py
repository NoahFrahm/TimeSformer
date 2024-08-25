# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""A More Flexible Video models."""

import torch
import torch.nn as nn
from timesformer.models import build_model
from torch.nn import functional as F
from einops import rearrange

import gc

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
                Mlp(in_features=self.num_experts * token_size, hidden_features=self.num_experts * token_size, out_features=num_experts),
                nn.Softmax(dim=1)
            )
            self.classification_head = Mlp(in_features=self.expert_dims, hidden_features=self.expert_dims, out_features=num_classes)
        elif self.out_method == 'concatenated': # MLP that processes concatenated input
            self.classification_head = Mlp(in_features=num_experts * self.expert_dims, hidden_features=num_experts * self.expert_dims, out_features=num_classes)

    
    def forward(self, x):
        # x -> (b x (n x d)) batch x tokens x token_dim
        B, N, D = x.shape
        
        # Routing
        x_tmp = rearrange(x, 'b n d-> (b n) d',b=B,n=N,d=D)
        S = F.softmax(torch.matmul(x_tmp, self.expert_embeddings.weight.t()), dim=1) # Token to expert score (b x n) x e
        S = rearrange(S, '(b n) e-> b n e',b=B,n=N,e=self.num_experts) # Token to expert score
        G, I = torch.topk(S, self.expert_capacity, dim=2) # Gating matrix b x e x k
        P = F.one_hot(I, num_classes=self.sequence_length) # Token gathering matrix (b x e x k x n)

        # Select Tokens
        P = torch.argmax(P, dim=-1)  # Converts to shape 5x4x2
        x_widened = x.unsqueeze(1).expand(-1, self.num_experts, -1, -1)
        selected_tokens = torch.gather(x_widened, 2, P.unsqueeze(-1).expand(-1, -1, -1, D))    
        selected_tokens = rearrange(selected_tokens, 'b e k d-> e b (k d)',b=B,e=self.num_experts,k=self.expert_capacity,d=D)

        # Apply experts
        expert_results = [expert(selected_tokens[expert_number]) for expert_number, expert in enumerate(self.experts)] # expert x (expert_capacity token_size)
        expert_results = torch.stack(expert_results, dim=1)
        # i have a list of 4 x 5 x 1536 tensors and a weight matrix of 5 x 4 i want to do a weighted sum on the tensors based on 

        # Final output creation
        if self.out_method == 'weighted':
            x = rearrange(x, 'b e d-> b (e d)',b=B,e=self.num_experts, d=D)
            weights = self.sum_weights(x)
            weights_reshaped = weights.unsqueeze(-1)
            weighted_sum = (expert_results * weights_reshaped).sum(dim=1)
            # weighted_sum = torch.einsum('i...,i->...', torch.stack(expert_results, dim=0), weights)
            return self.classification_head(weighted_sum)
        elif self.out_method == 'concatenated':
            # TODO: gotta fix this
            return self.classification_head(torch.cat(expert_results, dim=1))


class TimeMOE(nn.Module):
    """ mixture of expert across four modalities depth, pose, optical flow, rgb

        Args:
            cfg: contains all cfg specifications for the modality models to be built
            tokens: the number of tokens that will come from each modality model
    """

    # def __init__(self, cfg, topk, tokens, num_experts=12, classes=4):
    def __init__(self, cfg, tokens=768):
        super().__init__()

        # 4 Encoder models
        self.depth = build_model(cfg)
        self.depth.eval()
        self.pose = build_model(cfg)
        self.pose.eval()
        self.flow = build_model(cfg)
        self.flow.eval()
        self.rgb = build_model(cfg)
        self.rgb.eval()

        # moe block
        self.expert_mixture = ExpertChoice() # call build model with cfg here too?

    
    def forward(self, x):
        depth, pose, flow, rgb = x
        # depth, pose, flow, rgb = self.test_token, self.test_token, self.test_token, self.test_token
        
        with torch.no_grad():
            depth_tokens = self.depth(depth, get_feature=True)
    
        with torch.no_grad():
            pose_tokens = self.pose(pose, get_feature=True)
        
        with torch.no_grad():
            flow_tokens = self.flow(flow, get_feature=True)
    
        with torch.no_grad():
            rgb_tokens = self.rgb(rgb, get_feature=True)

        combined = torch.stack((depth_tokens, pose_tokens, flow_tokens, rgb_tokens), 0) # modality x batch x 768
        modality_tokens = combined.permute(1, 0, 2) # batch x modality x 768
        # modality_tokens = modality_tokens.flatten(start_dim=1) # batch x (modality * 768)

        expert_choice = self.expert_mixture(modality_tokens)

        return expert_choice


# # TODO: what can we modify to get prompt guidance but not lose pretraining weights?
# # load state dict strict=False so maybe not the biggest issue
# @MODEL_REGISTRY.register()
# class pgt_vit_base_patch16_224(nn.Module):
#     def __init__(self, cfg, **kwargs):
#         super(pgt_vit_base_patch16_224, self).__init__()
#         self.pretrained=True
#         patch_size = 16

#         self.model = PGTVisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

#         self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
#         self.model.default_cfg = default_cfgs['vit_base_patch16_224']
#         self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
#         pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
#         if self.pretrained:
#             load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

#     def forward(self, x):
#         x = self.model(x)
#         return x


# @MODEL_REGISTRY.register()
# class PGTTimeSformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
#         super(PGTTimeSformer, self).__init__()
#         self.pretrained=True
#         self.model = PGTVisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

#         self.attention_type = attention_type
#         self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
#         self.num_patches = (img_size // patch_size) * (img_size // patch_size)
#         if self.pretrained:
#             load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    
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


        