import wandb
import torch
import numpy as np
import time
from monai.losses import DiceLoss
from pytorch_lightning.callbacks import Callback

from utils import to_1hot, generate_contour_over_image, hausdorff_distance
from log_tools.log_utils import get_side_by_side


class WandbLoggerCallback(Callback):
    def __init__(self, project_name, log_dir, config) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(reduction="none")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        wandb.log({"train/loss": outputs['loss'], 
                   "train/seg_loss": outputs["seg_loss"], 
                   })
        # Load data from train dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape, t_idx = batch
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        sample_indices, self.subject_idx, gt_shape = sample_indices.squeeze(0), int(idx[0]), gt_shape.squeeze(0)
        t_idx = t_idx.squeeze(0).tolist()
        if pl_module.current_epoch % (pl_module.val_interval // 2) == 0 and 10 < self.subject_idx < 17:
            data_gt_shape = seg_values.reshape(*gt_shape[:2], -1, gt_shape[-1]).shape
            train_gt_shape = (*gt_shape[:2], data_gt_shape[2], len(t_idx))
            seg_values = seg_values.reshape((*gt_shape[:2], -1, gt_shape[-1]))[..., t_idx].reshape((-1, seg_values.shape[-1]))
            k_values = k_values.reshape((*gt_shape[:2], -1, gt_shape[-1], k_values.shape[-1]))[..., t_idx, :].reshape((-1, k_values.shape[-1]))
            img_values = img_values.reshape((*gt_shape[:2], -1, gt_shape[-1], img_values.shape[-1]))[..., t_idx, :].reshape((-1, img_values.shape[-1]))
            dec_coords = dec_coords.reshape((*gt_shape[:2], -1, gt_shape[-1], dec_coords.shape[-1]))[..., t_idx, :].reshape((-1, dec_coords.shape[-1]))
            gt_seg = seg_values.reshape(train_gt_shape).cpu().numpy()
            gt_img = img_values.reshape(train_gt_shape).cpu().numpy().__abs__()
            gt_kspace = k_values.reshape(train_gt_shape).cpu().numpy().__abs__()
            
            # If we are evaluating a volume, we only want to evaluate the middle slice
            self.if_evaluate_volume = train_gt_shape[2] != 1
            if self.if_evaluate_volume:
                z_idx = 4
                train_gt_shape = (*train_gt_shape[:2], 1, train_gt_shape[3])
                gt_seg = gt_seg[..., z_idx, :]
                gt_img = gt_img[..., z_idx, :]
                gt_kspace = gt_kspace[..., z_idx, :]
            
            # Evaluate the model
            pred_seg_values = outputs['seg']
            pred_seg = torch.argmax(pred_seg_values, dim=-1).numpy().reshape(train_gt_shape)
            side_by_side = get_side_by_side(gt_img, gt_seg, recon_pred=np.ones_like(gt_img), seg_pred=pred_seg, as_video=True)
            side_by_side_videos = wandb.Video(side_by_side.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            pred_seg_1hot = to_1hot(torch.argmax(pred_seg_values, dim=-1), num_class=pl_module.num_classes)[None].cpu().moveaxis(-1, 1)
            
            # Generate overlay videos
            overlay_pred_list, overlay_gt_list = generate_contour_over_image(gt_img, pred_seg, gt_seg)
            overlay_pred = np.transpose(np.stack(overlay_pred_list, axis=-1), (3, 2, 0, 1))
            overlay_gt = np.transpose(np.stack(overlay_gt_list, axis=-1), (3, 2, 0, 1))
            overlay_pred_video = wandb.Video(overlay_pred.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            overlay_gt_video = wandb.Video(overlay_gt.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            
            # Calculate metrics for segmentation and reconstruction
            gt_seg_tensor = torch.from_numpy(gt_seg).to(torch.float32).reshape(1, -1)
            gt_seg_1hot = to_1hot(gt_seg_tensor, num_class=pl_module.num_classes)
            dice = 1 - self.dice_loss(pred_seg_1hot.round(), gt_seg_1hot)
            dice_scores = dice.mean(0).squeeze().tolist()
            pred_set = to_1hot(torch.from_numpy(pred_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
            gt_set = to_1hot(torch.from_numpy(gt_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
            hd_scores = hausdorff_distance(pred_set, gt_set)
            
            # Log the results
            wandb.log({
                "train mri images": side_by_side_videos,
                "train pred images": overlay_pred_video,
                "train gt images": overlay_gt_video,
                "train/dice_LV_Pool": dice_scores[1],
                "train/dice_LV_Myo": dice_scores[2],
                "train/dice_RV_Pool": dice_scores[3],
                "train/hd_LV_Pool": hd_scores[1],
                "train/hd_LV_Myo": hd_scores[2],
                "train/hd_RV_Pool": hd_scores[3],
            })
            
            del side_by_side, side_by_side_videos
            return
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        pl_module.log_dict({"val/loss": outputs['loss']}) # for cheackpoint monitor
        wandb.log({"val/loss": outputs['loss'], 
                   "val/seg_loss": outputs["seg_loss"]
                   })
        # Load data from validation dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape, t_idx = batch
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        sample_indices, self.subject_idx, gt_shape = sample_indices.squeeze(0), int(idx[0]), gt_shape.squeeze(0)
        if self.subject_idx < 12:
            val_gt_shape = seg_values.reshape(*gt_shape[:2], -1, gt_shape[-1]).shape
            gt_seg = seg_values.reshape(val_gt_shape).cpu().numpy()
            gt_img = img_values.reshape(val_gt_shape).cpu().numpy().__abs__()
            gt_kspace = k_values.reshape(val_gt_shape).cpu().numpy().__abs__()
            
            # If we are evaluating a volume, we only want to evaluate the middle slice
            self.if_evaluate_volume = val_gt_shape[2] != 1
            if self.if_evaluate_volume:
                z_idx = 4
                val_gt_shape = (*gt_shape[:2], 1, gt_shape[-1])
                gt_seg = gt_seg[..., z_idx, :]
                gt_img = gt_img[..., z_idx, :]
                gt_kspace = gt_kspace[..., z_idx, :]
            
            # Evaluate the model
            pred_seg_values = pl_module.evaluate(input_coords, input_values, dec_coords, gt_shape=val_gt_shape, sample_indices=sample_indices, as_cpu=True)
            pred_seg = torch.argmax(pred_seg_values, dim=-1).numpy().reshape(val_gt_shape)
            side_by_side = get_side_by_side(gt_img, gt_seg, recon_pred=np.ones_like(gt_img), seg_pred=pred_seg, as_video=True)
            side_by_side_videos = wandb.Video(side_by_side.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            pred_seg_1hot = to_1hot(torch.argmax(pred_seg_values, dim=-1), num_class=pl_module.num_classes)[None].cpu().moveaxis(-1, 1)
            
            # Generate overlay videos
            overlay_pred_list, overlay_gt_list = generate_contour_over_image(gt_img, pred_seg, gt_seg)
            overlay_pred = np.transpose(np.stack(overlay_pred_list, axis=-1), (3, 2, 0, 1))
            overlay_gt = np.transpose(np.stack(overlay_gt_list, axis=-1), (3, 2, 0, 1))
            overlay_pred_video = wandb.Video(overlay_pred.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            overlay_gt_video = wandb.Video(overlay_gt.astype(np.uint8), caption=f"Subject {self.subject_idx}")
            
            # Calculate metrics for segmentation and reconstruction
            gt_seg_tensor = torch.from_numpy(gt_seg).to(torch.float32).reshape(1, -1)
            gt_seg_1hot = to_1hot(gt_seg_tensor, num_class=pl_module.num_classes)
            dice = 1 - self.dice_loss(pred_seg_1hot.round(), gt_seg_1hot)
            dice_scores = dice.mean(0).squeeze().tolist()
            pred_set = to_1hot(torch.from_numpy(pred_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
            gt_set = to_1hot(torch.from_numpy(gt_seg), num_class=pl_module.num_classes).squeeze(3).moveaxis([1, 3], [0, 1])
            hd_scores = hausdorff_distance(pred_set, gt_set)

            # Log the results
            wandb.log({
                "val mri images": side_by_side_videos,
                "val pred images": overlay_pred_video,
                "val gt images": overlay_gt_video,
                "val/dice_LV_Pool": dice_scores[1],
                "val/dice_LV_Myo": dice_scores[2],
                "val/dice_RV_Pool": dice_scores[3],
                "val/hd_LV_Pool": hd_scores[1],
                "val/hd_LV_Myo": hd_scores[2],
                "val/hd_RV_Pool": hd_scores[3],
            })
            del side_by_side, side_by_side_videos
        return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, *args) -> None:
        # Load data from test dataset
        input_coords, input_values, sample_indices, idx, dec_coords, seg_values, img_values, k_values, gt_shape, t_idx = batch
        input_coords, input_values, dec_coords = input_coords.squeeze(0), input_values.squeeze(0), dec_coords.squeeze(0)
        seg_values, img_values, k_values = seg_values.squeeze(0), img_values.squeeze(0), k_values.squeeze(0)
        sample_indices, self.subject_idx, gt_shape = sample_indices.squeeze(0), int(idx[0]), gt_shape.squeeze(0)
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
        side_by_side_videos = wandb.Video(side_by_side.astype(np.uint8), caption=f"Subject {self.subject_idx}")
        pred_seg_1hot = to_1hot(torch.argmax(pred_seg_values, dim=-1), num_class=pl_module.num_classes)[None].cpu().moveaxis(-1, 1)
        
        # Generate overlay videos
        overlay_pred_list, overlay_gt_list = generate_contour_over_image(gt_img, pred_seg, gt_seg)
        overlay_pred = np.transpose(np.stack(overlay_pred_list, axis=-1), (3, 2, 0, 1))
        overlay_gt = np.transpose(np.stack(overlay_gt_list, axis=-1), (3, 2, 0, 1))
        overlay_pred_video = wandb.Video(overlay_pred.astype(np.uint8), caption=f"Subject {self.subject_idx}")
        overlay_gt_video = wandb.Video(overlay_gt.astype(np.uint8), caption=f"Subject {self.subject_idx}")
        
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