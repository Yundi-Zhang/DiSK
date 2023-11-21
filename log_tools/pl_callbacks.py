import wandb
import torch
import numpy as np
import time
from monai.losses import DiceLoss
from pytorch_lightning.callbacks import Callback

from log_tools.log_utils import get_side_by_side
from utils import to_1hot, generate_contour_over_image, hausdorff_distance


class WandbLoggerCallback(Callback):
    def __init__(self, log_dir, config) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(reduction="none")
        self.train_epoch_start_time = None
        self.val_epoch_start_time = None
        self.test_epoch_start_time = None
        self.test_dice_scores = []
        self.test_hd_scores = []
        self.test_gt_im = []
        self.test_gt_seg = []
        self.test_pred_seg = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        # Get rid of image tensors
        log_dict = {f"train/{k}": v for k, v in outputs.items() if not isinstance(v, torch.Tensor) or len(v.shape) <= 1}
        # Convert any tensor into a scalar
        log_dict = {k: v if not isinstance(v, torch.Tensor) else v.detach().cpu().item() for k, v in log_dict.items()}
        pl_module.log_dict(log_dict)

        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape = batch
        subject_idx = int(idx[0])
        if pl_module.current_epoch % (pl_module.val_interval // 2) == 0 and 10 < subject_idx < 17:

            # Recreate GT im and seg arrays
            seg_values, img_values = seg_values.squeeze(0), img_values.squeeze(0)
            gt_shape = tuple(gt_shape.squeeze(0))
            gt_seg = seg_values.reshape(gt_shape).cpu().numpy()
            gt_img = img_values.reshape(gt_shape).cpu().numpy().__abs__()

            # Get segmentation predicted by model
            pred_seg_ = outputs['seg']
            pred_labels_ = torch.argmax(pred_seg_, dim=-1)
            pred_labels = pred_labels_.reshape(gt_shape).detach().cpu().numpy()

            # Zero-fill K-space
            input_values, sample_indices = input_values.squeeze(0), sample_indices.squeeze(0)
            input_k = torch.zeros(gt_shape, dtype=torch.complex64)
            sample_indices = sample_indices.cpu().numpy()
            input_k[tuple(sample_indices)] = input_values[:, 0].cpu()
            input_k = input_k.cpu().numpy()
            # Generate overlay videos
            video = get_side_by_side(gt_img, gt_seg, input_k, pred_labels, num_classes=pred_seg_.shape[-1])
            video = wandb.Video(video, caption=f"Subject_{subject_idx}")

            # Log video
            wandb.log({
                f"train_videos/subject_{subject_idx}": video,
            })
            del video
            return
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        # Get rid of image tensors
        log_dict = {f"val/{k}": v for k, v in outputs.items() if not isinstance(v, torch.Tensor) or len(v.shape) <= 1}
        # Convert any tensor into a scalar
        log_dict = {k: v if not isinstance(v, torch.Tensor) else v.detach().cpu().item() for k, v in log_dict.items()}

        # Load data from validation dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape = batch
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        sample_indices, subject_idx, gt_shape = sample_indices.squeeze(0), idx[0], gt_shape.squeeze(0)

        gt_shape = tuple(gt_shape)
        # Calculate metrics for segmentation
        pred_seg_ = outputs['seg']
        pred_labels_ = torch.argmax(pred_seg_, dim=-1)
        pred_set = to_1hot(pred_labels_, num_class=pred_seg_.shape[-1]).reshape((*gt_shape, pred_seg_.shape[-1]))
        gt_set = to_1hot(seg_values.squeeze(-1), num_class=pred_seg_.shape[-1]).reshape((*gt_shape, pred_seg_.shape[-1]))
        # Hausdorff distance metric
        hd_scores = hausdorff_distance(pred_set.permute(-1, -2, 0, 1), gt_set.permute(-1, -2, 0, 1))

        # Log metrics
        metrics = {"val/hd_LV_Pool": hd_scores[1],
                   "val/hd_LV_Myo": hd_scores[2],
                   "val/hd_RV_Pool": hd_scores[3],
                   "val/hd_FG": hd_scores[1:].max(),
                   }
        log_dict = {**log_dict, **metrics}
        pl_module.log_dict(log_dict)

        # Log videos
        subject_idx = int(subject_idx)
        if subject_idx < 12:
            # Recreate GT im and seg arrays
            gt_seg = seg_values.reshape(gt_shape).cpu().numpy()
            gt_img = img_values.reshape(gt_shape).cpu().numpy().__abs__()
            pred_labels = pred_labels_.reshape(gt_shape).detach().cpu().numpy()

            # Zero-fill K-space
            input_values, sample_indices = input_values.squeeze(0), sample_indices.squeeze(0)
            input_k = torch.zeros(gt_shape, dtype=torch.complex64)
            sample_indices = sample_indices.cpu().numpy()
            input_k[tuple(sample_indices)] = input_values[:, 0].cpu()
            input_k = input_k.cpu().numpy()
            # Generate overlay videos
            video = get_side_by_side(gt_img, gt_seg, input_k, pred_labels, num_classes=pred_seg_.shape[-1])
            video = wandb.Video(video, caption=f"Subject_{subject_idx}")

            # Log video
            wandb.log({
                f"val_videos/subject_{subject_idx}": video,
            })
            del video
        return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        # Get rid of image tensors
        log_dict = {f"test/{k}": v for k, v in outputs.items() if not isinstance(v, torch.Tensor) or len(v.shape) <= 1}
        # Convert any tensor into a scalar
        log_dict = {k: v if not isinstance(v, torch.Tensor) else v.detach().cpu().item() for k, v in log_dict.items()}
        
        # Load data from test dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape = batch
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        sample_indices, subject_idx, gt_shape = sample_indices.squeeze(0), idx[0], gt_shape.squeeze(0)
        
        # Evaluate the model
        pred_seg_ = pl_module.evaluate(input_coords, input_values, dec_coords, gt_shape=gt_shape, as_cpu=True)
        gt_shape = tuple(gt_shape)
        # Calculate metrics for segmentation
        pred_labels_ = torch.argmax(pred_seg_, dim=-1)
        pred_set = to_1hot(pred_labels_, num_class=pred_seg_.shape[-1]).reshape((*gt_shape, pred_seg_.shape[-1]))
        gt_set = to_1hot(seg_values.squeeze(-1), num_class=pred_seg_.shape[-1]).reshape((*gt_shape, pred_seg_.shape[-1]))
        # Hausdorff distance metric
        hd_scores = hausdorff_distance(pred_set.permute(-1, -2, 0, 1), gt_set.permute(-1, -2, 0, 1))
        
        # Log metrics
        metrics = {"test/hd_LV_Pool": hd_scores[1],
                   "test/hd_LV_Myo": hd_scores[2],
                   "test/hd_RV_Pool": hd_scores[3],
                   "test/hd_FG": hd_scores[1:].max(),
                   }
        log_dict = {**log_dict, **metrics}
        pl_module.log_dict(log_dict)
        
        # Log videos
        # Recreate GT im and seg arrays
        gt_seg = seg_values.reshape(gt_shape).cpu().numpy()
        gt_img = img_values.reshape(gt_shape).cpu().numpy().__abs__()
        pred_labels = pred_labels_.reshape(gt_shape).detach().cpu().numpy()
        # Zero-fill K-space
        input_values, sample_indices = input_values.squeeze(0), sample_indices.squeeze(0)
        input_k = torch.zeros(gt_shape, dtype=torch.complex64)
        sample_indices = sample_indices.cpu().numpy()
        input_k[tuple(sample_indices)] = input_values[:, 0].cpu()
        input_k = input_k.cpu().numpy()
        # Generate overlay videos
        video = get_side_by_side(gt_img, gt_seg, input_k, pred_labels, num_classes=pred_seg_.shape[-1])
        video = wandb.Video(video, caption=f"Subject_{subject_idx}")
        wandb.log({
            f"test_videos/subject_{subject_idx}": video,
        })
        del video
        
        # Save dice socres and hausdorff distances
        dice_scores = outputs["dice_scores"].squeeze(-1)
        self.test_dice_scores.append(dice_scores)
        self.test_hd_scores.append(hd_scores)
        self.test_gt_im.append(gt_img)
        self.test_gt_seg.append(gt_seg)
        self.test_pred_seg.append(pred_labels)
        return
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.train_epoch_start_time
        wandb.log({"train_epoch_duration": epoch_duration})
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_duration = time.time() - self.val_epoch_start_time
        wandb.log({"val_epoch_duration": epoch_duration})
    
    def on_test_epoch_start(self, trainer, pl_module):
        self.test_epoch_start_time = time.time()
        
    def on_test_epoch_end(self, trainer, pl_module):
        # Save averaged Dice scores, maximal Hausdorff distances, and gt/pred images over test set
        all_dice = torch.stack(self.test_dice_scores, dim=0)
        all_dice = all_dice.numpy()
        all_hd = torch.stack(self.test_hd_scores, dim=0)
        all_hd = all_hd.numpy()
        im_gts = self.test_gt_im
        seg_gts = self.test_gt_seg
        seg_preds = self.test_pred_seg
        pl_module.test_results = {"dice": all_dice,
                                  "hd": all_hd,
                                  "im_gt": im_gts,
                                  "seg_gt": seg_gts,
                                  "seg_pred": seg_preds,
                                  }
        epoch_duration = time.time() - self.test_epoch_start_time
        wandb.log({"test_epoch_duration": epoch_duration})
        return 