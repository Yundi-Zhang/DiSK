from typing import Tuple, Dict, Any, List, Union, Optional
import torch
import pytorch_lightning as pl
from torch import optim
from torch import optim, nn
from monai.losses import DiceLoss
import numpy as np

from model.layers import Relu
from model.encoder import PerceiverEncoder
from model.decoder import CrossAttentionDecoder, SegmentationHead
from model.pos_encoding import POS_ENCODING_LUT
from utils import to_1hot


class KTSTModule(pl.LightningModule):
    def __init__(self, train_dataset, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_interval = kwargs.get("check_val_every_n_epoch")
        self.lr = float(kwargs.get("lr"))

        # Build model architecture
        self.layer_class = Relu
        sample_coords, sample_values, *_ = train_dataset[0]
        self.coord_dims = sample_coords.shape[-1]
        self.non_channel_dims = tuple((0, *list(range(2, len(sample_values.shape)))))  # (0: batch, 1: channel, 2: dims)
        self.input_is_complex = sample_values.dtype == torch.complex64
        self.input_pixel_size = 2 if self.input_is_complex else 1

        # Define positional encoding for encoder
        enc_embed = POS_ENCODING_LUT[kwargs["enc_embedding"]]
        self.enc_coord_embedding = enc_embed(in_dim=self.coord_dims,
                                             coords_freq_scale=kwargs.get("enc_freq_scale"),
                                             gauss_num_frequencies=kwargs.get("enc_gauss_num_freqs"),
                                             nerf_num_frequencies=kwargs.get("enc_nerf_num_freqs"),
                                             cape_K=kwargs.get("enc_att_token_size"),
                                             **kwargs)

        self.enc_value_embedding = enc_embed(in_dim=self.input_pixel_size,
                                             coords_freq_scale=kwargs.get("enc_freq_scale"),
                                             gauss_num_frequencies=kwargs.get("enc_gauss_num_freqs"),
                                             nerf_num_frequencies=kwargs.get("enc_nerf_num_freqs")[:self.input_pixel_size],
                                             cape_K=kwargs.get("enc_att_token_size"),
                                             **kwargs)

        # Define positional encoding for decoder
        dec_embed = POS_ENCODING_LUT[kwargs["dec_embedding"]]
        self.dec_coord_embedding = dec_embed(in_dim=self.coord_dims,
                                             coords_freq_scale=kwargs.get("dec_freq_scale"),
                                             gauss_num_frequencies=kwargs.get("dec_gauss_num_freqs"),
                                             nerf_num_frequencies=kwargs.get("dec_nerf_num_freqs"),
                                             cape_K=kwargs.get("enc_att_token_size"),
                                             **kwargs)
        
        # Define encoder
        enc_num_hidden_layers = kwargs['enc_num_hidden_layers']
        enc_hidden_size = kwargs['enc_hidden_size']
        self.encoder = PerceiverEncoder(self.enc_coord_embedding.out_dim,
                                        self.enc_value_embedding.out_dim,
                                        enc_hidden_size,
                                        self.layer_class,
                                        num_hidden_layers=enc_num_hidden_layers,
                                        **kwargs)
        
        # Define decoder
        dec_num_hidden_layers = kwargs['dec_num_hidden_layers']
        dec_hidden_size = kwargs['dec_hidden_size']
        self.decoder = CrossAttentionDecoder(self.dec_coord_embedding.out_dim,
                                             self.encoder.out_size,
                                             dec_hidden_size,
                                             self.layer_class,
                                             num_hidden_layers=dec_num_hidden_layers,
                                             **kwargs)
        self.seg_head = SegmentationHead(input_size=self.decoder.out_size, layer_num=1, **kwargs)
        
        # Define loss
        self.enc_weight_reg = float(kwargs.get("enc_weight_reg", 0.0))
        self.dec_weight_reg = float(kwargs.get("dec_weight_reg", 0.0))
        self.bce_loss = nn.BCELoss(reduction="none")
        self.dice_loss = DiceLoss(reduction="none")
        self._use_seg_bce = kwargs.get("use_seg_bce", True)
        self.num_classes = self.train_dataset.num_classes
        
        # Save the test results
        self.test_results = dict()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self,
                enc_coords: torch.Tensor,
                enc_values: torch.Tensor,
                dec_coords: torch.Tensor,
                **kwargs):
        latent_codes = self.forward_encoder(enc_coords, enc_values, **kwargs)
        seg = self.forward_decoder(dec_coords, latent_codes, **kwargs)
        return seg
    
    def forward_encoder(self,
                        coords: torch.Tensor,
                        values: torch.Tensor,
                        **kwargs):
        if self.input_is_complex:
            values = torch.view_as_real(values).squeeze(1)
        values = self.enc_value_embedding(values)
        coords = self.enc_coord_embedding(coords)
        latent_codes = self.encoder(coords, values, **kwargs)
        return latent_codes
    
    def forward_decoder(self,
                        dec_coords: torch.Tensor,
                        latent_codes: torch.Tensor,
                        gt_shape: torch.Tensor,
                        **kwargs):
        dec_coords = self.dec_coord_embedding(dec_coords)
        decoder_output = self.decoder(dec_coords, latent_codes, gt_shape=gt_shape, **kwargs)
        seg = self.seg_head(decoder_output)
        return seg

    def segmentation_criterion(self, pred: Union[torch.Tensor, List[torch.Tensor]], gt: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seg_values_1hot = to_1hot(gt[..., 0], num_class=self.num_classes)
        # Take the mean segmentation loss, leaving the class dimension untouched
        seg_dice_loss = self.dice_loss(pred, seg_values_1hot).mean(self.non_channel_dims)  # MONAI Dice loss requires a batch dimension
        dice = 1 - seg_dice_loss
        seg_dice_loss = seg_dice_loss.mean()
        return_dict = {"dice_LV_pool": dice[1],
                       "dice_LV_myo": dice[2],
                       "dice_RV_pool": dice[3],
                       "dice_FG": dice[1:].mean(),
                       "seg_dice_loss": seg_dice_loss
                       }
        seg_loss = seg_dice_loss
        if self._use_seg_bce:
            seg_bce_loss = self.bce_loss(pred, seg_values_1hot).mean(self.non_channel_dims)
            seg_bce_loss = seg_bce_loss.mean()
            seg_loss += seg_bce_loss
            return_dict["seg_bce_loss"] = seg_bce_loss

        return_dict["seg_loss"] = seg_loss
        return seg_loss, return_dict, dice
    
    def regularization_criterion(self, latent_codes: Union[torch.Tensor, List[torch.Tensor]]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dict = dict()
        if isinstance(latent_codes, list):
            latent_codes = torch.cat(latent_codes, dim=0)
        reg_loss = torch.zeros((1,), dtype=torch.float32, device=latent_codes.device)
        if self.enc_weight_reg:
            enc_reg_loss = sum((p * p).sum() for p in self.encoder.parameters()) * self.enc_weight_reg
            reg_loss += enc_reg_loss
            loss_dict["enc_reg_loss"] = enc_reg_loss
        if self.dec_weight_reg:
            dec_reg_loss = sum((p * p).sum() for p in self.decoder.parameters()) * self.dec_weight_reg
            reg_loss += dec_reg_loss
            loss_dict["dec_reg_loss"] = dec_reg_loss
        loss_dict["reg_loss"] = reg_loss
        return reg_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape = batch
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        img_values, sample_indices = img_values.squeeze(0), sample_indices.squeeze(0)
        gt_shape = tuple(gt_shape.squeeze(0).tolist())

        # Forward pass of encoder
        latent_codes = self.forward_encoder(input_coords, input_values, sample_indices=sample_indices, gt_shape=gt_shape)
        
        # Forward pass of decoder
        data_gt_shape = seg_values.reshape((*gt_shape[:2], -1, gt_shape[-1])).shape
        seg = self.forward_decoder(dec_coords, latent_codes, gt_shape=data_gt_shape)
        
        # Compute loss
        return_dict = dict()
        loss = torch.zeros((1,), dtype=torch.float32, device=input_coords.device)
        seg_loss, seg_loss_dict, dice = self.segmentation_criterion(seg.permute(1, 0)[None], seg_values)
        loss += seg_loss
        return_dict = {**return_dict, **seg_loss_dict}
        reg_loss, reg_loss_dict = self.regularization_criterion(latent_codes)  # compute regularization loss
        loss += reg_loss
        return_dict = {**return_dict, **reg_loss_dict}

        return {'loss': loss, 
                **return_dict, 
                'seg': seg.detach().cpu(), 
                'dice_scores': dice.detach().cpu(),
                }

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    @torch.no_grad()
    def evaluate(self,
                 encoder_coords: torch.Tensor,
                 encoder_values: torch.Tensor,
                 decoder_coords: torch.Tensor,
                 as_cpu: bool = False,
                 as_numpy: bool = False,
                 **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        assert decoder_coords.shape[-1] == self.coord_dims
        assert encoder_coords is not None
        assert encoder_values is not None
        assert encoder_coords.shape[0] == encoder_values.shape[0]
        if self.encoder.parameters().__next__().device == "cuda":
            encoder_coords, encoder_values, decoder_coords = encoder_coords.cuda(), encoder_values.cuda(), decoder_coords.cuda()
        pred_seg = self.forward(encoder_coords, encoder_values, decoder_coords, **kwargs)
        if as_numpy:
            return pred_seg.cpu().numpy()
        if as_cpu:
            return pred_seg.cpu()
        return pred_seg