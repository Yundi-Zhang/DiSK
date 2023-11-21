import torch
from torch import nn
from typing import Union, List
from model.layers import Layer, CrossAttentionLayer


class SegmentationHead(nn.Module):
    def __init__(self, input_size, num_classes=4, layer_num=1, **kwargs):
        super(SegmentationHead, self).__init__()
        la = [nn.Linear(input_size, input_size) for _ in range(layer_num - 1)]
        la += [nn.Linear(input_size, num_classes)]
        self.layers = nn.Sequential(*la)
        self.out_size = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            out = [torch.softmax(self.layers(i), dim=1) for i in x]
        else:
            out = torch.softmax(self.layers(x), dim=1)
        return out
    
    
class CrossAttentionDecoder(nn.Module):

    def __init__(self, coord_size: int, enc_out_size: int, hidden_size: int, layer_class: Layer, num_hidden_layers: int, value_size: int = None,
                 **kwargs):
        super(CrossAttentionDecoder, self).__init__()
        self.att_heads = kwargs["dec_att_num_heads"]

        a = [CrossAttentionLayer(enc_out_size,
                                 enc_out_size,
                                 self.att_heads,
                                 dim_feedforward=hidden_size,
                                 activation_class=layer_class,
                                 batch_first=True,  # NOTE THIS IS NOT DEFAULT
                                 **kwargs)
             for i in range(num_hidden_layers)]
        self.ca_hid_layers = nn.ModuleList(a)
        if value_size is None:
            value_size = 0
        self.needs_padding = coord_size + value_size != enc_out_size
        self.info_token_width = enc_out_size
        self.out_size = enc_out_size

    def forward(self,
                dec_coords: torch.Tensor,
                info_tokens: Union[torch.Tensor, List[torch.Tensor]],
                **kwargs) -> torch.Tensor:

        if self.needs_padding:
            dec_tokens = torch.zeros((dec_coords.shape[0], self.info_token_width), device=dec_coords.device)
            dec_tokens[:, :dec_coords.shape[1]] = dec_coords
        else:
            dec_tokens = dec_coords

        # Allow for different info tokens at each layer (ex. each encoder layer output)
        if not isinstance(info_tokens, list):
            info_tokens = [info_tokens] * len(self.ca_hid_layers)

        for ca_layer, info_t in zip(self.ca_hid_layers, info_tokens):
            dec_tokens = dec_tokens + ca_layer(dec_tokens, info_t)
        return dec_tokens
