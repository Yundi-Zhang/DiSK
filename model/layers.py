import abc
from typing import Optional
import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class Relu(Layer):
    LUT_NAME = "relu"

    def __init__(self, in_size, out_size, **kwargs):
        super(Relu, self).__init__(in_size, out_size, **kwargs)
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class SelfAttentionLayer(nn.Module):
    r"""
    -- TAKEN FROM nn.TransformerEncoderLayer AND MODIFIED TO ACCEPT OUR CUSTOM ACTIVATION LAYERS --

    TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation_class: custom callable layer class (see model/layers.py).
            These typically include a linear layer followed by an activation function.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, d_out: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(SelfAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = activation_class(d_model, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, encoding_tokens: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            encoding_tokens: the token set to the encoder layer (required).
            attn_mask: a 2D or 3D mask preventing attention to certain positions.
                Must be of shape (L,S) or (N⋅num_heads,L,S), where N is the batch size,
                L is the target sequence length, and S is the source sequence length.
                A 2D mask will be broadcasted across the batch while a 3D mask allows for a
                different mask for each entry in the batch.
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            key_padding_mask: a mask of shape (N,S) indicating which elements within key to ignore for
                the purpose of attention (i.e. treat as “padding”). For unbatched query, shape should be (S).
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the
                corresponding key value will be ignored for the purpose of attention.
                For a float mask, it will be directly added to the corresponding key value.

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = encoding_tokens
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))  # TODO: Might wanna remove skip connection to allow for variable ouput size

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    r"""
    -- TAKEN FROM nn.TransformerEncoderLayer AND MODIFIED TO ACCEPT OUR CUSTOM ACTIVATION LAYERS --

    CrossAttentionLayer is made up of attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation_class: custom callable layer class (see model/layers.py).
            These typically include a linear layer followed by an activation function.
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, d_out: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation_class: Layer = Relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 **kwargs) -> None:
        factory_kwargs = {'device': kwargs.get("device", None), 'dtype': kwargs.get("dtype", None)}
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = activation_class(d_model, dim_feedforward, **kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_out, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_out, eps=layer_norm_eps, **factory_kwargs)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def forward(self, decoding_tokens: torch.Tensor, info_tokens: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the decoder layer.

        Args:
            decoding_tokens: tokens to be processed by the decoder (required).
                These will be the embedded reconstruction coordinates.
            info_tokens: tokens meant to introduce information to the decoding process (required).
                These will be the encoder's output tokens (separate input coordinates passed by the encoder).
            attn_mask: a 2D or 3D mask preventing attention to certain positions.
                Must be of shape (L,S) or (N⋅num_heads,L,S), where N is the batch size,
                L is the target sequence length, and S is the source sequence length.
                A 2D mask will be broadcasted across the batch while a 3D mask allows for a
                different mask for each entry in the batch.
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the corresponding position is not allowed to attend.
                For a float mask, the mask values will be added to the attention weight.
                If both attn_mask and key_padding_mask are supplied, their types should match.
            key_padding_mask: a mask of shape (N,S) indicating which elements within key to ignore for
                the purpose of attention (i.e. treat as “padding”). For unbatched query, shape should be (S).
                Binary and float masks are supported.
                For a binary mask, a True value indicates that the
                corresponding key value will be ignored for the purpose of attention.
                For a float mask, it will be directly added to the corresponding key value.

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = decoding_tokens
        if self.norm_first:
            x = x + self._ca_block(self.norm1(x), self.norm1(info_tokens), attn_mask, key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._ca_block(x, info_tokens, attn_mask, key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # attention block
    def _ca_block(self, decode_tokens: torch.Tensor, info_tokens: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.cross_attn(decode_tokens, info_tokens, info_tokens,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        return self.dropout(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return self.dropout(x)
    