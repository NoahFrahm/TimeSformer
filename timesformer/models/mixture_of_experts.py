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
import math

# https://huggingface.co/blog/AviSoori1x/seemoe
# https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
# https://github.com/google/flaxformer/blob/main/flaxformer/architectures/moe/routing.py#L647



# Transformer positional encoding module
class PositionalEncoding(nn.Module):
    # sourced from: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
            Args:
            d_model:      dimension of embeddings
            dropout:      randomly zeroes-out some of the input
            max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()

        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)

        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)

        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)

        # calc divisor for positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)

        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)

        # add dimension
        pe = pe.unsqueeze(0)

        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x:   embeddings (batch_size, seq_length, d_model)
        Returns: embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        x = x.permute(1, 0, 2) #NOTE: added this so it works with transformer encoder layer?

        # perform dropout
        return self.dropout(x)

# Transformer that takes in input directly
class FusionTransformer(nn.Module):
    """
    Sequence length is capped at 5000 tokens
    """
    def __init__(self,
                 trg_classes, # this is class number
                 d_model,
                 dropout,
                 n_head,
                 dim_feedforward,
                 n_layers,
                ):
        super().__init__()

        # set the dimensions for positional encoding (the total number of inputs (i.e. token size)) and the dropout
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Only using Encoder of Transformer model
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.d_model = d_model
        # decodes the transformer output into the 4 classes
        self.decoder = nn.Linear(d_model, trg_classes) #TODO: maybe swap this out with non linear layer?

    def forward(self, x, get_features=False):
        # here we pass the input sequence to the positional encoding layer to add positional encoding to each token
        x_emb = self.positional_encoding(x)
        # Shape (output) -> (Sequence length, batch size, d_model)
        output = self.transformer_encoder(x_emb)
        # We want our output to be in the shape of (batch size, d_model) so that
        # we can use it with CrossEntropyLoss hence averaging using first (Sequence length) dimension
        # Shape (mean) -> (batch size, d_model)
        # Shape (decoder) -> (batch size, d_model)
        if get_features: return output.mean(0)
        return self.decoder(output.mean(0)) #TODO: why do we do mean?
        


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
    
    def __init__(self, input_tokens=4, token_size=768, capacity_factor=2, num_experts=4, num_classes=4, out_method='weighted', expert_setup='mlp'):
        super().__init__()

        self.num_experts = num_experts
        self.out_method = out_method
        self.sequence_length = input_tokens
        self.expert_capacity = int((input_tokens * capacity_factor) / num_experts) # number of tokens each expert can take
        self.expert_dims = self.expert_capacity * token_size

        # breakpoint()

        # Expert embedding matrix (Wg)
        self.expert_embeddings = nn.Embedding(num_embeddings=num_experts, embedding_dim=token_size)

        # Expert models
        if expert_setup == 'mlp':
            self.experts = nn.ModuleList([
                Expert(self.expert_dims, out_features=self.expert_dims) for _ in range(num_experts)
            ])
        elif expert_setup == 'transformer':
            # TODO: we must somehow map the original positional encoding back to the expert output to preserve that information? maybe not also
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

    def __init__(self, cfg, fusion_head, rgb_model, pose_model, depth_model, flow_model):
        super().__init__()
 
        # 4 Encoder models 
        self.depth, self.depth_finetune = depth_model
        self.pose, self.pose_finetune = pose_model
        self.flow, self.flow_finetune = flow_model
        self.rgb, self.rgb_finetune = rgb_model

        # fusion head
        self.fusion_head, self.fusion_finetune = fusion_head

        self.all_tokens = cfg.ALL_TOKENS
        self.fusion_method = 'transformer' #TODO: figure this out

    
    def forward(self, x):
        depth, pose, flow, rgb = x

        # breakpoint()
        if self.depth_finetune:depth_tokens = self.depth(depth, get_feature=True, get_all=self.all_tokens)
        else: 
            with torch.no_grad(): depth_tokens = self.depth(depth, get_feature=True, get_all=self.all_tokens)

        # breakpoint()
        if self.pose_finetune: pose_tokens = self.pose(pose, get_feature=True, get_all=self.all_tokens)
        else:
            with torch.no_grad(): pose_tokens = self.pose(pose, get_feature=True, get_all=self.all_tokens)

        # breakpoint()
        if self.flow_finetune: flow_tokens = self.flow(flow, get_feature=True, get_all=self.all_tokens)
        else:
            with torch.no_grad(): flow_tokens = self.flow(flow, get_feature=True, get_all=self.all_tokens)

        # breakpoint()
        if self.rgb_finetune: rgb_tokens = self.rgb(rgb, get_feature=True, get_all=self.all_tokens)
        else:
            with torch.no_grad(): rgb_tokens = self.rgb(rgb, get_feature=True, get_all=self.all_tokens)

        combined = torch.stack((depth_tokens, pose_tokens, flow_tokens, rgb_tokens), 0) # modality x batch x 768
        
        if self.fusion_method == 'all':
            modality_tokens = combined.permute(1, 0, 2, 3) # batch x modality x token number x 768
        elif self.fusion_method == 'transformer':
            modality_tokens = combined.permute(1, 0, 2)
        else:
            modality_tokens = combined.permute(1, 0, 2) # batch x modality x 768

        # cfg.transformer
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

# This helper gives the necessary info for loading modality models
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
    """
    MOE with MLP experts
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.pretrained=cfg.PRETRAINED

        self.model = ExpertChoice(input_tokens=cfg.MODEL.INPUT_TOKENS, token_size=cfg.MODEL.TOKEN_SIZE, capacity_factor=2, num_experts=cfg.MODEL.NUM_EXPERTS, num_classes=cfg.MODEL.NUM_CLASSES, out_method=cfg.MODEL.OUT_METHOD, expert_setup='mlp')

        if self.pretrained:
            ...

       
    def forward(self, x):
        x = self.model(x)
        return x


@MODEL_REGISTRY.register()
class moe_transformer_fusion(nn.Module):
    """
    MOE but instead of MLP experts we use transformer experts
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.pretrained=cfg.PRETRAINED

        self.model = ExpertChoice(input_tokens=cfg.MODEL.INPUT_TOKENS, token_size=cfg.MODEL.TOKEN_SIZE, capacity_factor=2, num_experts=cfg.MODEL.NUM_EXPERTS, num_classes=cfg.MODEL.NUM_CLASSES, out_method=cfg.MODEL.OUT_METHOD, expert_setup='transformer')

        if self.pretrained:
            ...

        
    def forward(self, x):
        x = self.model(x)
        return x


@MODEL_REGISTRY.register()
class transformer_fusion(nn.Module):
    """
    One transformer as fusion head
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.pretrained=cfg.PRETRAINED

        self.model = FusionTransformer(
                trg_classes=cfg.MODEL.NUM_CLASSES,
                d_model=cfg.MODEL.TOKEN_SIZE,
                dropout=cfg.MODEL.DROPOUT,
                n_head=cfg.MODEL.NUM_HEADS,
                dim_feedforward=cfg.MODEL.TOKEN_SIZE * 4,
                n_layers=cfg.MODEL.NUM_LAYERS)

        if self.pretrained:
            ...

        
    def forward(self, x):
        x = self.model(x)
        return x


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
        self.model = MultiModalityTimeS(cfg, **sub_models)
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
