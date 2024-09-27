

import copy
from typing import Optional, Any, Union, Callable

import torch
import os
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, save_path=None, layer_num=0) -> None:
        super(TransformerEncoderLayerWithWeights, self).__init__(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first,
            device, dtype)
        self.save_path=save_path
        self.layer_num=layer_num
        

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            # TODO: ensure that this attention weight order is correct
            # attention, attention_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            # x = x + attention
            x = x + self._ff_block(self.norm2(x))
        else:
            # attention_weights, attention = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            # x = self.norm1(x + attention)
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Returns attention weights
        """
        x, attention_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        if self.save_path:
            weight_save_path = os.path.join(self.save_path, 'layer_num' + self.layer_num + '.pt')
            torch.save(attention_weights, weight_save_path)

        return self.dropout1(x)



# create version of fusion head that saves attention maps at each attention step
# or allow them to be visualzed and saved at each step (easiest to implement)
# then create a special instance of fusion head that uses this transformer setup instead
# call inference with this and it should save attention map outputs
# figure out how to visualize them...