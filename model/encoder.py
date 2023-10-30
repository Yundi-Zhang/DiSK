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
        max_set_size = kwargs.get("enc_att_max_set_size", -1)
        if isinstance(max_set_size, str):
            max_set_size = eval(max_set_size)
        assert isinstance(max_set_size, int)
        self.max_set_size = max_set_size

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
        self.needs_padding = coord_size + in_features != token_size
        self.info_token_width = token_size

    def forward(self, coords: torch.Tensor, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        cat_input = torch.cat((coords, x), dim=-1)
        if self.needs_padding:
            info_tokens = torch.zeros((coords.shape[0], self.info_token_width), device=coords.device)
            info_tokens[:, :cat_input.shape[1]] = cat_input
        else:
            info_tokens = cat_input
        x = self.pos_encoding
        outs = []
        for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers):
            x = x + ca_layer(x, info_tokens)
            x = x + sa_layer(x)
            outs.append(x)
        return outs