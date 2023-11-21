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

        # Log images/videos
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

    # TODO: Refactor
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        # Load data from test dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape, t_idx = batch
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        sample_indices, subject_idx, gt_shape = sample_indices.squeeze(0), int(idx[0]), gt_shape.squeeze(0)
        test_gt_shape = seg_values.reshape(*gt_shape[:2], -1, gt_shape[-1]).shape
        gt_seg = seg_values.reshape(test_gt_shape).cpu().numpy()
        gt_img = img_values.reshape(test_gt_shape).cpu().numpy().__abs__()
        gt_kspace = k_values.reshape(test_gt_shape).cpu().numpy().__abs__()
        
        # If we are evaluating a volume, we only want to evaluate the middle slice
        self.if_evaluate_volume = test_gt_shape[2] != 1
        if self.if_evaluate_volume:
            z_idx = 4
            test_gt_shape = (*gt_shape[:2], 1, gt_shape[-1])
            gt_seg = gt_seg[..., z_idx, :]
            gt_img = gt_img[..., z_idx, :]
            gt_kspace = gt_kspace[..., z_idx, :]
        
        # Evaluate the model
        pred_seg_values = pl_module.evaluate(input_coords, input_values, dec_coords, gt_shape=test_gt_shape, sample_indices=sample_indices, as_cpu=True)
        pred_seg = torch.argmax(pred_seg_values, dim=-1).numpy().reshape(test_gt_shape)
        side_by_side = get_side_by_side(gt_img, gt_seg, recon_pred=np.ones_like(gt_img), seg_pred=pred_seg, as_video=True)
        side_by_side_videos = wandb.Video(side_by_side.astype(np.uint8), caption=f"Subject {subject_idx}")
        pred_seg_1hot = to_1hot(torch.argmax(pred_seg_values, dim=-1), num_class=pl_module.num_classes)[None].cpu().moveaxis(-1, 1)
        
        # Generate overlay videos
        overlay_pred_list, overlay_gt_list = generate_contour_over_image(gt_img, pred_seg, gt_seg)
        overlay_pred = np.transpose(np.stack(overlay_pred_list, axis=-1), (3, 2, 0, 1))
        overlay_gt = np.transpose(np.stack(overlay_gt_list, axis=-1), (3, 2, 0, 1))
        overlay_pred_video = wandb.Video(overlay_pred.astype(np.uint8), caption=f"Subject {subject_idx}")
        overlay_gt_video = wandb.Video(overlay_gt.astype(np.uint8), caption=f"Subject {subject_idx}")
        
        # Calculate metrics for segmentation and reconstruction
        gt_seg_tensor = torch.from_numpy(gt_seg).to(torch.float32).reshape(1, -1)
        gt_seg_1hot = to_1hot(gt_seg_tensor, num_class=pl_module.num_classes)
        dice = 1 - self.dice_loss(pred_seg_1hot.round(), gt_seg_1hot)
        dice_scores = dice.mean(0).squeeze()
        pred_set = to_1hot(torch.from_numpy(pred_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
        gt_set = to_1hot(torch.from_numpy(gt_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
        hd_scores = hausdorff_distance(pred_set, gt_set)
        
        # Log the results
        wandb.log({
            "test mri images": side_by_side_videos,
            "test pred images": overlay_pred_video,
            "test gt images": overlay_gt_video,
            "test/dice_LV_Pool": dice_scores[1],
            "test/dice_LV_Myo": dice_scores[2],
            "test/dice_RV_Pool": dice_scores[3],
            "test/hd_LV_Pool": hd_scores[1],
            "test/hd_LV_Myo": hd_scores[2],
            "test/hd_RV_Pool": hd_scores[3],
        })
        del side_by_side, side_by_side_videos
    
        # Scores for evaluation
        pl_module.eval_dice_scores.append(dice_scores)
        pl_module.eval_hd_scores.append(hd_scores)
        pl_module.eval_pred_segs.append(pred_seg)
        pl_module.eval_gt_segs.append(gt_seg)
        pl_module.eval_gt_ims.append(gt_img)
        
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
        all_dice_scores = torch.stack(pl_module.eval_dice_scores, dim=0)
        avg_dice_scores = all_dice_scores.mean(dim=0)
        all_hd_scores = torch.stack(pl_module.eval_hd_scores, dim=0)
        avg_hd_scores = all_hd_scores.mean(dim=0)
        all_seg_preds = pl_module.eval_pred_segs[-5:]
        all_seg_gts = pl_module.eval_gt_segs[-5:]
        all_im_gts = pl_module.eval_gt_ims[-5:]
        
        pl_module.results = {"dice_scores": avg_dice_scores,
                             "hausdorff_distances": avg_hd_scores,
                             "seg_pred": all_seg_preds,
                             "seg_gt": all_seg_gts,
                             "im": all_im_gts}
        epoch_duration = time.time() - self.test_epoch_start_time
        wandb.log({"test_epoch_duration": epoch_duration})
        return 