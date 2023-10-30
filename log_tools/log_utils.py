import numpy as np
from utils import normalize_image


def get_side_by_side(gt_img: np.ndarray,
                     gt_seg: np.ndarray,
                     recon_pred: np.ndarray=None,
                     seg_pred: np.ndarray=None, 
                     kspace_pred: np.ndarray=None, 
                     gt_kspace: np.ndarray=None,
                     kspace_pred_ift: np.ndarray=None,
                     template_img: np.ndarray=None,
                     template_k: np.ndarray=None,
                     as_image: bool=False,
                     as_video: bool=False):
    if not as_image and not as_video:
        assert False, "Either as_image or as_video must be True."
    if as_image and as_video:
        assert False, "Only one of as_image and as_video can be True."
    
    if as_image: # Only choose one time frame to show
        gt_seg, gt_img = gt_seg[..., 0, 0], gt_img[..., 0, 0]
        seg_pred = seg_pred[..., 0, 0] if seg_pred is not None else None
        recon_pred = recon_pred[..., 0, 0] if recon_pred is not None else None
        kspace_pred = kspace_pred[..., 0, 0] if kspace_pred is not None else None
        gt_kspace = gt_kspace[..., 0, 0] if gt_kspace is not None else None
        kspace_pred_ift = kspace_pred_ift[..., 0, 0] if kspace_pred_ift is not None else None
        template_img = template_img[..., 0, 0] if template_img is not None else None
        template_k = template_k[..., 0, 0] if template_k is not None else None
    
    im_list = []
    if template_img is None and template_k is None:
        if recon_pred is not None:
            recon_ims = np.concatenate((gt_img, recon_pred), axis=0)
            recon_ims = normalize_image(recon_ims)
            im_list += [recon_ims]
        seg_ims = np.concatenate((gt_seg / 3.0, seg_pred / 3.0), axis=0)
        templ_im, templ_k = None, None
    else:
        templ_im = template_img if template_img is not None else np.ones_like(gt_img)
        templ_k = template_k if template_k is not None else np.ones_like(gt_img)
        recon_ims = np.concatenate((gt_img, recon_pred, templ_im), axis=0)
        recon_ims = normalize_image(recon_ims)
        seg_ims = np.concatenate((gt_seg / 3.0, seg_pred / 3.0, np.ones_like(gt_img)), axis=0)
        im_list += [recon_ims]
        
    im_list += [seg_ims]
    
    if kspace_pred is not None and gt_kspace is not None:
        kspace_ims_ = [np.log(gt_kspace + 1.0e-8), np.log(kspace_pred + 1.0e-8)]
        if templ_k is not None:
            kspace_ims_ += [np.log(templ_k + 1.0e-8)]
        kspace_ims = np.concatenate(kspace_ims_, axis=0)
        kspace_ims = normalize_image(kspace_ims)
        im_list += [kspace_ims]
        
    if kspace_pred_ift is not None:
        ift_ims_ = [np.ones_like(gt_img), normalize_image(kspace_pred_ift)]
        if template_img is not None and template_k is not None:
            ift_ims_ += [np.ones_like(gt_img)]
        ift_ims = np.concatenate(ift_ims_, axis=0)
        im_list += [ift_ims]

    side_by_side_ims = np.concatenate(im_list, axis=1)
    if as_image:
        return side_by_side_ims
    if as_video:
        side_by_side_videos = np.transpose(side_by_side_ims, (3, 2, 0, 1)) * 255
        return side_by_side_videos