import os
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom


class SliceDataSet(Dataset):
    def __init__(self, root_dir=None, list_dir=None, transform=None, img_size=(320, 320)):
        self.root_dir = os.path.join(root_dir, "slice_data")
        self.sample_list = open(os.path.join(list_dir, "slice_train.txt")).readlines()
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip("\n")

        h5f = h5py.File(os.path.join(self.root_dir, sample_name + ".h5"), "r")
        image = h5f["image"][:].astype(np.float32)
        label = h5f["label"][:].astype(np.float32)  # (H, W)
        salience = h5f["salience"][:].astype(np.float32)  # (3, H, W)
        h5f.close()

        x, y = image.shape
        if x != self.img_size[0] or y != self.img_size[1]:
            image = zoom(image, (self.img_size[0] / x, self.img_size[1] / y), order=3)
            salience = zoom(salience, (1, self.img_size[0] / x, self.img_size[1] / y), order=3)
            label = zoom(label, (self.img_size[0] / x, self.img_size[1] / y), order=0)

        sample = {
            "image": image,
            "label": label,
            "salience": salience,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    


class VolumeDataSet(Dataset):
    def __init__(self, root_dir=None, list_dir=None, split="val", transform=None, img_size=(320, 320)):
        self.volume_dir = os.path.join(root_dir, "vol_data")
        self.salience_dir = os.path.join(root_dir, "vol_salience")
        self.sample_list = open(
            os.path.join(list_dir, f"vol_{split}.txt")
        ).readlines()
        self.transform = transform
        self.output_size = img_size

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip("\n")

        h5f = h5py.File(os.path.join(self.volume_dir, sample_name + ".h5"), "r")
        image = h5f["image"][:].astype(np.float32)
        label = h5f["label"][:].astype(np.float32)  # (H, W, D)
        h5f.close()

        h5f = h5py.File(os.path.join(self.salience_dir, sample_name + ".h5"), "r")
        salience = h5f["salience"][:].astype(np.float32)  # (3, H, W, D)
        h5f.close()

        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            salience = zoom(salience, (1, self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        
        sample = {
            "image": image,
            "label": label,
            "salience": salience,
        }

        if self.transform:
            sample = self.transform(sample)

        sample["name"] = sample_name
        return sample



class RandomRotFlip(object):
    def __call__(self, sample):
        image, label, salience = sample["image"], sample["label"], sample["salience"]

        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            salience = np.rot90(salience, k, axes=(1, 2)).copy()

        elif random.random() > 0.5:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            salience = np.flip(salience, axis=axis + 1).copy()

        return {"image": image, "label": label, "salience": salience}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, salience = sample["image"], sample["label"], sample["salience"]
        return {
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            "label": torch.from_numpy(label.astype(np.float32)).unsqueeze(0),
            "salience": torch.from_numpy(salience.astype(np.float32)),
        }
