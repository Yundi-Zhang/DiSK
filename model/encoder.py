import torch
from torch import nn
from typing import Union, List

from model.layers import Layer, SelfAttentionLayer, CrossAttentionLayer


class PerceiverEncoder(nn.Module):

    def __init__(self, coord_size: int, in_features: int, hidden_size: int, layer_class: Layer, num_hidden_layers: int,
                 latent_nodes: int = 128, **kwargs):
        super(PerceiverEncoder, self).__init__()
        self.att_heads = kwargs["enc_att_num_heads"]
        token_size = kwargs["enc_att_token_size"]

        a = [SelfAttentionLayer(token_size,
                                token_size,
                                self.att_heads,
                                dim_feedforward=hidden_size,
                                activation_class=layer_class,
                                batch_first=True,  # NOTE THIS IS NOT DEFAULT
                                **kwargs)
             for i in range(num_hidden_layers)]
        
        self.sa_layers = nn.ModuleList(a)
        self.pos_encoding = nn.Parameter(torch.randn(latent_nodes, token_size))
        a = [CrossAttentionLayer(token_size,
                                 token_size,
                                 self.att_heads,
                                 dim_feedforward=hidden_size,
                                 activation_class=layer_class,
                                 batch_first=True,  # NOTE THIS IS NOT DEFAULT
                                 **kwargs)
             for i in range(num_hidden_layers)]
        self.ca_layers = nn.ModuleList(a)
        self.out_size = token_size
        self.coord_needs_padding = coord_size != token_size
        self.val_needs_padding = in_features != token_size
        self.info_token_width = token_size

    def forward(self,
                coords: Union[torch.Tensor, List[torch.Tensor]],
                values: Union[torch.Tensor, List[torch.Tensor]],
                **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.coord_needs_padding:
            coords_ = torch.zeros((coords.shape[0], self.info_token_width), device=coords.device)
            coords_[:, :coords.shape[1]] = coords
            coords = coords_
        if self.val_needs_padding:
            values_ = torch.zeros((values.shape[0], self.info_token_width), device=values.device)
            values_[:, :values.shape[1]] = values
            values = values_
        info_tokens = values + coords

        x = self.pos_encoding
        outs = []
        for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
            x = x + ca_layer(x, info_tokens)
            x = x + sa_layer(x)
            outs.append(x)
        return outs