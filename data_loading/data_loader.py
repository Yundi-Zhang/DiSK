from typing import Iterable, Dict, Any, Tuple, Optional, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pickle
import math

from data_loading.data_utils import SimulateCartesian, NormalLines_Sampler, find_sax_images
from utils import normalize_image, to_1hot, fft2c_mri, ifft2c_mri

GETITEM_RETURN_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class AbstractDataset(Dataset):
    LUT_NAME = "abstract"

    def __init__(self, load_dir, pickle_name="dataset.pkl", **kwargs):
        self.load_dir = load_dir
        self.use_bboxes = kwargs.get("use_bboxes", True)
        self.z_seg_relative = kwargs.get("z_seg_relative", 4)
        self.cache_data = kwargs.get("cache_data", False)
        if self.cache_data:
            print("Loading dataset into RAM...")
        try:
            with open(pickle_name, 'rb') as handle:
                self.im_paths, self.seg_paths, self.bboxes = pickle.load(handle)
        except FileNotFoundError:
            self.im_paths, self.seg_paths, self.bboxes = self.find_images(**kwargs)
            with open(pickle_name, 'wb') as handle:
                pickle.dump([self.im_paths, self.seg_paths, self.bboxes], handle, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(self.im_paths) > 0

        # If a side_length value was left as -1 by the user, use the max shape of that dimension instead
        self.num_classes = self.load_nifti(0, z_seg_relative=self.z_seg_relative)[1].max() + 1
        # Sampling patterns
        self.sample_type = kwargs.get("sample_type", "random")
        self.acceleration = kwargs.get("acceleration", 4)
        self.complex_transformer = SimulateCartesian()
        self.coord_noise_std = kwargs.get("coord_noise_std", 0.0)
        assert self.coord_noise_std >= 0.0

    def __len__(self):
        return len(self.im_paths)

    def find_images(self, **kwargs):
        return find_sax_images(self.load_dir, **kwargs)

    def get_bboxes(self):
        return None

    def __getitem__(self, image_index) -> GETITEM_RETURN_TYPE:
        raise NotImplementedError
    
    def load_and_undersample(self, idx: int,
                             acc: Optional[int] = None,
                             cache_data: Optional[bool] = False,
                             **kwarwgs):
        # Load image and seg data
        if cache_data:
            assert self.ram_im_list is not None and self.ram_seg_list is not None, "RAM version only supports 2DplusT data right now."
            im_data, seg = self.ram_im_list[idx], self.ram_seg_list[idx]
        else:
            im_data, seg = self.load_nifti(idx, self.z_seg_relative)
        
        # Generate complex images and k-space data
        k_space_4D, img_complex_4D = self.generate_kdata_and_complex_image(im_data)
        
        # Undersample points based on given sampling pattern
        coords, sample_nd_indices, k_space_values = self.undersample_points(k_space_4D, acc)
        return seg, img_complex_4D, k_space_4D, coords, sample_nd_indices, k_space_values

    def load_nifti(self, img_idx: int,
                   z_seg_relative: int,
                   *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param img_idx: Index of image in dataset image list.
        :param z_seg_relative: Picked SAX slice relative to the start of the LV.
        """
        nii_img = nib.load(self.im_paths[img_idx])
        nii_seg = nib.load(self.seg_paths[img_idx])
        raw_shape = nii_img.shape

        # Should be in the form [Y1, X1, ..., Y2, X2, ...]
        if not self.use_bboxes:  # If no bbox passed, use entire image (i.e. 0 -> -1  indices for every image)
            bbox = tuple([0] * len(raw_shape) + [*raw_shape])
        else:
            assert self.bboxes is not None, "You have no bboxes, boi."
            bbox = self.bboxes[img_idx]
        assert len(bbox) % 2 == 0
        # If no bbox provided for a dimension, fill with range 0 to max_shape (i.e. entire image)
        if len(bbox) // 2 == 2:
            bbox = (*bbox[:2], 0, *bbox[-2:], raw_shape[2])
        if len(bbox) // 2 == 3:
            bbox = (*bbox[:3], 0, *bbox[-3:], raw_shape[3])
        idx_slices = (slice(bbox[0], bbox[0 + len(bbox)//2]),
                      slice(bbox[1], bbox[1 + len(bbox)//2]),
                      )

        frame_seg_data = nii_seg.dataobj[idx_slices].astype(np.uint8)
        z_seg_start = (frame_seg_data[..., 0] == 1).any((0, 1)).argmax()
        z = z_seg_start + z_seg_relative
        frame_seg_data = frame_seg_data[:, :, z]

        # Remove slice dimension and keep only 2D+time
        frame_im_data = nii_img.dataobj[idx_slices[0], idx_slices[1], z]
        assert len(frame_im_data.shape) == 3, f"Img path: {self.im_paths[img_idx]}"
        # Mirror image on x=y so the RV is pointing left (standard short axis view)
        frame_im_data = np.transpose(frame_im_data, (1, 0, 2))
        frame_seg_data = np.transpose(frame_seg_data, (1, 0, 2))
        return frame_im_data, frame_seg_data

    def generate_kdata_and_complex_image(self, frame_im_data: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
        """
        Add synthetic phase information into image data and simulate k-space measurements.
        """
        img = normalize_image(frame_im_data.astype(np.float32), scale=math.sqrt(2))
        img_ = np.transpose(img)
        img_complex_ = self.complex_transformer(img_)
        img_complex = np.transpose(img_complex_)
        k_space = fft2c_mri(torch.from_numpy(img_complex)).numpy()
        return k_space, img_complex
    
    def undersample_points(self, k_space: np.ndarray, acc=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Samples points from the kspace, image, and segmentation volume based on given sampling pattern. """
        try:
            acc = self.acceleration if acc is None else acc
            sampler = NormalLines_Sampler(acc=acc, data_shape=k_space.shape)
        except KeyError:
            raise ValueError("Unknown sampling pattern.")
        # Generate undersampling mask
        mask = sampler.get_mask() if acc > 1 else np.ones_like(k_space, dtype=np.int64)
        nd_indices = np.where(mask)
        # Sample points from the kspace, image, and segmentation volume
        sampled_nd_indices = np.stack(nd_indices, axis=-1)
        coords = sampled_nd_indices / np.array(k_space.shape)
        k_space_values = k_space[nd_indices][:, None]
        return coords, nd_indices, k_space_values

    @property
    def add_coord_noise(self) -> bool:
        return self.coord_noise_std > 0.0

    def apply_coord_noise(self, coords: np.ndarray) -> np.ndarray:
        if self.add_coord_noise:
            coords += np.random.normal(0.0, self.coord_noise_std, coords.shape)
        return coords


class Seg2DplusT_SAX_Kspace(AbstractDataset):

    def __init__(self, *args, **kwargs):
        super(Seg2DplusT_SAX_Kspace, self).__init__(*args, **kwargs)
        # Load images and segmentations into RAM to accelerate the process
        self.ram_im_list = []
        self.ram_seg_list = []
        if self.cache_data:
            for i in range(self.__len__()):
                frame_im_data, frame_seg_data = self.load_nifti(i, self.z_seg_relative)
                frame_im_data_normalized = normalize_image(frame_im_data) * 255.
                self.ram_im_list.append(frame_im_data_normalized.astype(np.uint8))
                self.ram_seg_list.append(frame_seg_data.astype(np.uint8))
        
    def __getitem__(self, idx: int) -> GETITEM_RETURN_TYPE:
        return self.generate_item(idx)

    def generate_item(self, idx: int, acc: Optional[int] = None):
        seg_vol, img_complex_vol, k_space_vol, sample_coords, sample_nd_indices, sample_k_space = \
            self.load_and_undersample(idx, acc, use_bbox=self.use_bboxes, cache_data=self.cache_data)
        sample_nd_indices = np.array(sample_nd_indices)

        # Get decoder coords, image, and seg values (for entire image)
        dec_nd_indices = torch.meshgrid(torch.arange(img_complex_vol.shape[0], dtype=torch.long),
                                        torch.arange(img_complex_vol.shape[1], dtype=torch.long),
                                        torch.arange(img_complex_vol.shape[-1], dtype=torch.long),
                                        )
        dec_nd_indices = tuple([i.reshape(-1).numpy() for i in dec_nd_indices])
        dec_coords = np.stack(dec_nd_indices, axis=-1) / np.array(img_complex_vol.shape)
        img_values = img_complex_vol[dec_nd_indices][:, None]
        k_values = k_space_vol[dec_nd_indices][:, None]
        seg_values = seg_vol[dec_nd_indices][:, None]

        # Apply coord noise
        sample_coords = self.apply_coord_noise(sample_coords)
        dec_coords = self.apply_coord_noise(dec_coords)

        # Convert to tensors
        sample_coords = torch.from_numpy(sample_coords).to(torch.float32)
        sample_k_space = torch.from_numpy(sample_k_space).to(torch.complex64)
        sample_nd_indices = torch.from_numpy(sample_nd_indices).to(torch.long)
        idx = torch.tensor(idx)
        dec_coords = torch.from_numpy(dec_coords).to(torch.float32)
        seg_values = torch.from_numpy(seg_values).to(torch.float32)
        img_values = torch.from_numpy(img_values).to(torch.complex64)
        k_values = torch.from_numpy(k_values).to(torch.complex64)
        gt_vol_shape = torch.tensor(k_space_vol.shape).to(torch.int64)
        return sample_coords, sample_k_space, sample_nd_indices, idx, \
            dec_coords, seg_values, img_values, k_values, gt_vol_shape

    
class Seg2DplusT_SAX_Kspace_test(Seg2DplusT_SAX_Kspace):

    @property
    def add_coord_noise(self):
        return False
