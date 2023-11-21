import numpy as np
from utils import normalize_image, ifft2c_mri, generate_contour_over_image


def get_side_by_side(gt_img: np.ndarray,
                     gt_seg: np.ndarray,
                     input_k: np.ndarray,
                     pred_labels: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """ Create a concatenate image array. Left-most column in k-space and image reconstruction.
    Middle column is GT image and predicted gray-scale segmentation. Right column is GT and pred
    contours overlayed on GT image.
    Returned array is of shape (time, channels, height, width) """
    col_list = []
    # Normalize K-space image to [0, 1)
    input_k_n = normalize_image(np.log(input_k.__abs__() + 1.0e-8))
    # Normalize inverse FT image to [0, 1)
    input_k_to_im = ifft2c_mri(input_k)
    input_k_to_im_n = normalize_image(input_k_to_im.__abs__())
    # Create k-space column
    input_ims = np.concatenate((input_k_n, input_k_to_im_n), axis=0)
    input_ims = np.stack([input_ims]*3, axis=-1)  # To RGB
    col_list.append(input_ims)

    # Create image + label column
    pred_labels_gray = pred_labels / num_classes
    gt_ims = np.concatenate((normalize_image(gt_img), pred_labels_gray), axis=0)
    gt_ims = np.stack([gt_ims]*3, axis=-1)  # To RGB
    col_list.append(gt_ims)

    # Create image with contour overlay column
    overlay_pred_list, overlay_gt_list = generate_contour_over_image(gt_img, pred_labels, gt_seg)
    seg_ims = np.concatenate((overlay_gt_list, overlay_pred_list), axis=0)
    col_list.append(seg_ims)

    # Concatenate columns together
    video = np.concatenate(col_list, axis=1)
    # Convert to (time, channels, H, W)
    video = np.transpose(video, (2, 3, 0, 1))
    # To uint8
    video = video * 255
    video = video.astype(np.uint8)
    return video
