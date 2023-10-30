from typing import Iterable, Dict, Any, Tuple, Optional, List, Union
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import cv2
import pickle
import math
import sys

from data_loading.data_utils import SimulateCartesian, NormalLines_Sampler, find_sax_images
from data_loading.augmentations import AbstractAug, TranslateCoords, RotateCoords, GammaShift
from utils import normalize_image, to_1hot, fft2c_mri, ifft2c_mri

GETITEM_RETURN_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor, torch.Tensor]


KSPACE_GETITEM_RETURN_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class AbstractDataset(Dataset):
    LUT_NAME = "abstract"

    def __init__(self, load_dir, augmentations=(), pickle_name="dataset.pkl", **kwargs):
        self.coord_noise_std = kwargs.get("coord_noise_std", 0.0)
        assert self.coord_noise_std >= 0.0
        self.load_dir = load_dir
        self.use_bboxes = kwargs.get("use_bboxes", True)
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

        sample_size = kwargs.get("sample_size", 2048)  # TODO: Unused
        if isinstance(sample_size, str):
            sample_size = eval(sample_size)
        assert isinstance(sample_size, int)
        self.sample_size = sample_size

        self.x_ho_rate = kwargs.get("x_holdout_rate", 1)
        self.y_ho_rate = kwargs.get("y_holdout_rate", 1)
        self.z_ho_rate = kwargs.get("z_holdout_rate", 1)
        self.t_ho_rate = kwargs.get("t_holdout_rate", 1)
        # If a side_length value was left as -1 by the user, use the max shape of that dimension instead
        self.augs = self.parse_augmentations(augmentations)
        self.augment = self._augment
        self.num_aug_params = sum([a.num_parameters for a in self.augs])
        self.num_classes = self.load_nifti(0)[1].max() + 1
        # Sampling patterns
        self.sample_type = kwargs.get("sample_type", "random")
        self.acceleration = kwargs.get("acceleration", 4)
        self.complex_transformer = SimulateCartesian()

    def __len__(self):
        return len(self.im_paths)

    def find_images(self, **kwargs):
        return find_sax_images(self.load_dir, **kwargs)

    def get_bboxes(self):
        return None

    def __getitem__(self, image_index) -> Union[GETITEM_RETURN_TYPE, KSPACE_GETITEM_RETURN_TYPE]:
        raise NotImplementedError
    
    def load_and_undersample(self, idx: int, acc: Optional[int] = None, use_bbox: Optional[bool]=True, cache_data: Optional[bool]=False):
        # Load image and seg data
        if cache_data:
            assert self.ram_im_list is not None and self.ram_seg_list is not None, "RAM version only supports 2DplusT data right now."
            im_data_4D, seg_4D = self.ram_im_list[idx], self.ram_seg_list[idx]
        else:
            im_data_4D, seg_4D, _ = self.load_nifti(idx, use_bbox=use_bbox)
        
        # Generate complex images and k-space data
        k_space_4D, img_complex_4D = self.generate_kdata_and_complex_image(im_data_4D)
        
        # Undersample points based on given sampling pattern
        coords, sample_nd_indices, k_space_values = self.undersample_points(k_space_4D, acc)
        return seg_4D, img_complex_4D, k_space_4D, coords, sample_nd_indices, k_space_values
    
    def load_nifti(self,  img_idx: int, t: Optional[float] = None, use_bbox: bool = True, *args, **kwargs) \
            -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """Load image and segmentation files and undersample them according to hold-out rates.
        :param img_idx: Index of image in dataset image list.
        :param t: Specific time point for which to extract the (closest) 3D volume for.
                  Passing a time value here will use lazy loading to avoid loading the entire 4D array.
                  If no t value passed, the entire 4D array is returned.
        :param use_bbox: Whether to use precomputed bounding boxes (if self.bboxes is not None)
        """
        nii_img = nib.load(self.im_paths[img_idx])
        nii_seg = nib.load(self.seg_paths[img_idx])
        raw_shape = nii_img.shape

        # Should be in the form [Y1, X1, ..., Y2, X2, ...]
        if not use_bbox:  # If no bbox passed, use entire image (i.e. 0 -> -1  indices for every image)
            bbox = tuple([0] * len(nii_img.shape) + [-1] * len(nii_img.shape))
        else:
            assert self.bboxes is not None, "You have no bboxes, boi."
            bbox = self.bboxes[img_idx]
        assert len(bbox) % 2 == 0
        # If no bbox provided for a dimension, fill with range 0 to -1 (i.e. entire image)
        if len(bbox) // 2 == 2:
            bbox = (*bbox[:2], 0, *bbox[-2:], nii_img.shape[2])
        if len(bbox) // 2 == 3:
            bbox = (*bbox[:3], 0, *bbox[-3:], nii_img.shape[3])
        ho_rate = (self.x_ho_rate, self.y_ho_rate, self.z_ho_rate, self.t_ho_rate)
        idx_slices = tuple(slice(bbox[i], bbox[i + len(bbox)//2], step)
                           for i, step in zip(range(0, len(bbox)//2), ho_rate))

        # If t argument passed, extract only the preceding time frame closest the time float value
        used_t_idx = None
        if t is not None:
            assert 0.0 <= t < 1.0
            assert isinstance(t, float)
            # Find preceding time index closest to time value given
            used_t_idx = int(t * raw_shape[-1])
            used_t_idx -= used_t_idx % self.t_ho_rate
            if len(idx_slices) < 4:  # TODO: better way to detect this? Data could be 2d+t or >3D with no time axis
                print("Warning: You passed a 't' value in order to extract a specific time frame but your data "
                      "seems to have no time axis. Ignoring 't' argument...")
            else:
                # Replace time slice with a time slice of size 1
                idx_slices = (*idx_slices[:-1], slice(used_t_idx, used_t_idx+1, self.t_ho_rate))

        frame_im_data = nii_img.dataobj[idx_slices]
        frame_seg_data = nii_seg.dataobj[idx_slices].astype(np.uint8)
        assert len(frame_im_data.shape) == 4, f"Img path: {self.im_paths[img_idx]}"
        return frame_im_data, frame_seg_data, used_t_idx

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
        coords =  sampled_nd_indices / np.array(k_space.shape)
        k_space_values = k_space[nd_indices][:, None]
        return coords, nd_indices, k_space_values

    @property
    def _augment(self) -> bool:
        return len(self.augs) > 0

    @property
    def add_coord_noise(self) -> bool:
        return self.coord_noise_std > 0.0

    @staticmethod
    def parse_augmentations(augs: Iterable[Dict[str, Any]]) -> List[AbstractAug]:
        name_2_class = {"translation": TranslateCoords, "rotation": RotateCoords, "gamma": GammaShift}
        aug_instances = []
        for aug in augs:
            n, params = list(aug.items())[0]
            try:
                class_name = name_2_class[n]
            except KeyError:
                raise KeyError(f'Provided augmentation name "{n}" is not in the default dictionary of augmentations.')
            aug_instances.append(class_name(**params))
        return aug_instances

    def apply_augmentations(self, coords: np.ndarray, img: np.ndarray, seg: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
        aug_params = []
        if self.augment:
            data = {"coords": coords, "image": img, "seg": seg}
            for aug in self.augs:
                aug_params.extend(aug(data))
            coords, img, seg = data["coords"], data["image"], data["seg"]
        return coords, img, seg, aug_params

    def apply_coord_noise(self, coords: np.ndarray) -> np.ndarray:
        if self.add_coord_noise:
            coords += np.random.normal(0.0, self.coord_noise_std, coords.shape)
        return coords

    

class Seg2DplusT_SAX_Kspace(AbstractDataset):
    LUT_NAME = "cardiac_mri_kspace_2DplusT"

    def __init__(self, *args, **kwargs):
        super(Seg2DplusT_SAX_Kspace, self).__init__(*args, **kwargs)
        # Load images and segmentations into RAM to accelerate the process
        self.ram_im_list = []
        self.ram_seg_list = []
        if self.cache_data:
            for i in range(self.__len__()):
                frame_im_data, frame_seg_data, used_t_idx = self.load_nifti(i, use_bbox=self.use_bboxes)
                frame_im_data_normalized = normalize_image(frame_im_data) * 255.
                self.ram_im_list.append(frame_im_data_normalized.astype(np.uint8))
                self.ram_seg_list.append(frame_seg_data.astype(np.uint8))
        
    def __getitem__(self, idx: int) -> GETITEM_RETURN_TYPE:
        return self.generate_item(idx, z_idx=4)

    def generate_item(self, idx: int, z_idx: Optional[int] = None, t_idx: Optional[int] = None, acc: Optional[int] = None):
        seg_4D, img_complex_4D, k_space_4D, sample_coords, sample_nd_indices, sample_k_space = \
            self.load_and_undersample(idx, acc, use_bbox=self.use_bboxes, cache_data=self.cache_data)
        # Sample a single 2D slice from the 3D+t volume
        if z_idx is None:
            idx_slice = 4  # np.random.randint(0, frame_im_data_4D.shape[-2], 1)[0]
        else:
            idx_slice = z_idx
        if t_idx is None:
            # idx_frame = np.random.randint(0, img_complex_4D.shape[-1], 1)
            idx_frame = np.arange(0, img_complex_4D.shape[-1], 1)
        else:
            idx_frame = t_idx

        slice_indices = sample_nd_indices[2] == idx_slice
        sample_nd_indices = np.array(sample_nd_indices).T[slice_indices]
        sample_coords = sample_coords[slice_indices]
        sample_k_space = sample_k_space[slice_indices]

        # Get decoder coords, image, and seg values (for entire image)
        dec_nd_indices = torch.meshgrid(torch.arange(img_complex_4D.shape[0], dtype=torch.long),
                                        torch.arange(img_complex_4D.shape[1], dtype=torch.long),
                                        torch.tensor(idx_slice),
                                        torch.arange(img_complex_4D.shape[-1], dtype=torch.long),
                                        )
        dec_nd_indices = tuple([i.reshape(-1).numpy() for i in dec_nd_indices])
        dec_coords = np.stack(dec_nd_indices, axis=-1) / np.array(img_complex_4D.shape)
        img_values = img_complex_4D[dec_nd_indices][:, None]
        k_values = k_space_4D[dec_nd_indices][:, None]
        seg_values = seg_4D[dec_nd_indices][:, None]

        # Apply augmentations
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
        gt_vol_shape = torch.tensor(k_space_4D.shape).to(torch.int64)
        supervision_frame = torch.tensor(idx_frame).to(torch.long)
        return sample_coords, sample_k_space, sample_nd_indices, idx, dec_coords, seg_values, img_values, k_values, gt_vol_shape, supervision_frame

    
class Seg2DplusT_SAX_Kspace_test(Seg2DplusT_SAX_Kspace):
    @property
    def _augment(self):
        return False

    @property
    def add_coord_noise(self):
        return False