import os
import h5py
import random
import torch
import numpy as np
from torch.utils.data import Dataset


def random_rot_flip(image, label):
        # input shape: (H, W) or (H, W, D)
        k = np.random.randint(1, 4)
        image = np.rot90(image, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        return image, label


class RandomCrop3D(object):
    def __init__(self, img_size=(96, 96, 96)):
        self.crop_size = img_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        crop_size = self.crop_size

        if crop_size[0] > image.shape[0] or crop_size[1] > image.shape[1] or crop_size[2] > image.shape[2]:
            print("crop_size: {} is bigger than img_size: {}".format(crop_size, image.shape))

        h_start = random.randint(0, image.shape[0] - crop_size[0])
        w_start = random.randint(0, image.shape[1] - crop_size[1])
        d_start = random.randint(0, image.shape[2] - crop_size[2])
        image_croped = image[h_start: h_start + crop_size[0], w_start: w_start + crop_size[1], d_start: d_start + crop_size[2]]
        label_croped = label[h_start: h_start + crop_size[0], w_start: w_start + crop_size[1], d_start: d_start + crop_size[2]]
        sample = {'image': image_croped, 'label': label_croped}
        return sample


class RandomGenerator(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        sample = {'image': image, 'label': label}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        sample = {
            "image": torch.from_numpy(image).unsqueeze(0).float(),  # (1, H, W, D)
            "label": torch.from_numpy(label).unsqueeze(0).float()
        }
        return sample


class LiverVesselVolume(Dataset):
    def __init__(self, root_dir="../data/3Dircadb1/data_h5", list_dir="../list/3Dircadb1", split="train", transform=None):
        self.transform = transform
        with open(os.path.join(list_dir, split + ".txt")) as f:
            case_lst = f.readlines()
        self.sample_list = case_lst
        self.root_dir = root_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx].strip('\n')
        file_path = os.path.join(self.root_dir, case_name + ".h5")
        with h5py.File(file_path) as f:
            image_vol = f["image"][:]
            label_vol = f["label"][:]

        sample = {
            "image": image_vol.astype(np.float32), # (H, W, D)
            "label": label_vol.astype(np.float32)  # (H, W, D)
        }
        if self.transform:
            sample = self.transform(sample)
        sample["sample_name"] = case_name
        return sample
    

if __name__ == "__main__":
    from torchvision import transforms
    db_train = LiverVesselVolume(split="all")
    print(db_train.__len__())

    for i in range(db_train.__len__()):
        print(db_train[i]["label"].shape)