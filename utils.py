from typing import Union, Optional, Tuple
import numpy as np
import torch
import nibabel as nib
import nibabel.processing as nip
from torch import nn
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import hausdorff_distance as skhausdorff


def normalize_image(im: Union[np.ndarray, torch.Tensor], low: float = None, high: float = None, clip=True, scale: float=None) -> Union[np.ndarray, torch.Tensor]:
    """ Normalize array to range [0, 1] """
    if low is None:
        low = im.min()
    if high is None:
        high = im.max()
    if clip:
        im = im.clip(low, high)
    im_ = (im - low) / (high - low)
    if scale is not None:
        im_ = im_ * scale
    return im_


def to_1hot(class_indices: torch.Tensor, num_class) -> torch.FloatTensor:
    """ Converts index array to 1-hot structure. """
    seg = class_indices.to(torch.long).reshape((-1,))
    seg_1hot = torch.zeros((*seg.shape, num_class), dtype=torch.float32, device=class_indices.device)
    seg_1hot[torch.arange(0, seg.shape[0], dtype=torch.long), seg] = 1
    seg_1hot = seg_1hot.reshape((*class_indices.shape, num_class)).moveaxis(-1, 1)
    return seg_1hot


def resample_nib(img, voxel_spacing=(1, 1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing

    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation

    Returns:
    ----------
    new_img: The resampled nibabel image

    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2],
        shp[3] * zms[3] / voxel_spacing[3],
    ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp[:3], voxel_spacing[:3], new_shp[:3])
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img


class HDRLoss(nn.Module):
    def __init__(self, epsilon=1e-1, reduction="mean"):
        super().__init__()
        self.eps = epsilon
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.abs((x - y) / (x.detach() + self.eps))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("Unknown reduction: ", self.reduction)
        return loss
    
    
def ifft2c_mri(k):
    if isinstance(k, np.ndarray):
        x = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(0, 1)), norm='ortho', axes=(0, 1)), axes=(0, 1))
    elif isinstance(k, torch.Tensor):
        x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, dim=(0, 1)), norm='ortho', dim=(0, 1)), dim=(0, 1))
    else:
        raise ValueError("Not a numpy array or a torch tensor.")
    return x


def fft2c_mri(img, is_numpy=False):
    if is_numpy:
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(0, 1)), norm='ortho', axes=(0, 1)), axes=(0, 1))
    else:
        k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, dim=(0, 1)), norm='ortho', dim=(0, 1)), dim=(0, 1))
    return k


def overlay(seg: np.ndarray, image: np.ndarray, overlay_filename: Optional[str] = None):
    """Create overlay image of segmentation on top of the image."""
    seg_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) if len(seg.shape) == 3 else seg
    image = cv2.merge([image, image, image]) if len(image.shape) != 3 else image
    # cv2.imwrite(str(overlay_filename), image)
    colors = [(255, 255, 0), (51,255,153)]
    # Myo
    th = np.where(seg_gray == 2, 1, 0)
    contours, hierarchy = cv2.findContours(th.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colors[0], 1)
    # RV
    th = np.where(seg_gray == 3, 1, 0)
    contours, hierarchy = cv2.findContours(th.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colors[1], 1)
    if overlay_filename is not None:
        cv2.imwrite(str(overlay_filename), image)
    return image


def generate_contour_over_image(image: np.ndarray, seg: np.ndarray, gt_seg: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    overlay_pred_list = []
    overlay_gt_list = []
    for frame in range(seg.shape[-1]):
        frame_pred = seg[..., frame]
        frame_gt = gt_seg[..., frame]
        frame_im = image[..., frame]
        overlay_pred = overlay(frame_pred, normalize_image(frame_im, scale=255.0)).astype(np.uint8)
        overlay_gt = overlay(frame_gt, normalize_image(frame_im, scale=255.0)).astype(np.uint8)
        overlay_pred_list.append(overlay_pred)
        overlay_gt_list.append(overlay_gt)
    overlay_pred_arr = np.stack(overlay_pred_list, axis=2)
    overlay_gt_arr = np.stack(overlay_gt_list, axis=2)
    # Normalize arrays back to [0, 1)
    overlay_pred_arr = normalize_image(overlay_pred_arr)
    overlay_gt_arr = normalize_image(overlay_gt_arr)
    return overlay_pred_arr, overlay_gt_arr


def ssim(pred, gt, channel_axis=2, data_range=1.0):
    """
    Calculates the structural similarity index measure (SSIM) between two images. The images are in the shape of (H, W, C, T).
    """
    (ssim_scores, diff) = skssim(pred, gt, channel_axis=channel_axis, data_range=data_range, full=True)
    return ssim_scores


def hausdorff_distance(pred: torch.Tensor, gt: torch.Tensor):
    """
    Calculates the Hausdorff distance. pred is in the shape of (C, T H, W), and gt is in the shape of (C, T, H, W). T is the number of time frames, C is the number of classes, H is the height, and W is the width.
    """
    assert pred.shape == gt.shape, "The two sets must have the same shape."
    hd = torch.empty(pred.shape[:2])
    for c in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            hd[c, t] = skhausdorff(pred[c][t].detach().cpu().numpy(), gt[c][t].cpu().numpy())
    return hd.mean(dim=1)
