# Copyright 2020 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
from torch.utils.data.dataset import Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from skimage import io, transform
from PIL import Image
import os
import numpy as np
import torch
import random
import cv2


class AutoToonDataset(Dataset):
    def __init__(self, root='.', train=True, train_split=0.9, random_flip=True, always_flip=False, color_jitter=True, random_crop=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5072122 , 0.4338291 , 0.38466746], 
                                                                                           [0.08460502, 0.07533269, 0.07269038])])):
        self.train = train
        self.random_flip = random_flip
        self.always_flip = always_flip
        self.color_jitter = color_jitter
        self.random_crop = random_crop
        self.photos_root = os.path.join(root, 'photos')
        self.cari_root = os.path.join(root, 'caricatures')
        self.warp16_root = os.path.join(root, 'warp16')
        self.warp32_root = os.path.join(root, 'warp32')
        self.transform = transform

        self.photos = [os.path.join(self.photos_root, p) for p in os.listdir(self.photos_root) if os.path.isfile(os.path.join(self.photos_root, p)) and '.DS_Store' not in p]
        self.caris = [os.path.join(self.cari_root, c) for c in os.listdir(self.cari_root) if os.path.isfile(os.path.join(self.cari_root, c)) and '.DS_Store' not in c]
        self.warps16 = [os.path.join(self.warp16_root, w) for w in os.listdir(self.warp16_root) if os.path.isfile(os.path.join(self.warp16_root, w)) and '.DS_Store' not in w]
        self.warps32 = [os.path.join(self.warp32_root, w) for w in os.listdir(self.warp32_root) if os.path.isfile(os.path.join(self.warp32_root, w)) and '.DS_Store' not in w]
        self.photos = sorted(self.photos)  # sort names in ascending order to ensure deterministic behavior
        self.caris = sorted(self.caris)
        self.warps16 = sorted(self.warps16)
        self.warps32 = sorted(self.warps32)

        num_partitions = int(1. / (1 - train_split))
        self.data_pairs = [(p, c, w16, w32) for p, c, w16, w32 in zip(self.photos, self.caris, self.warps16, self.warps32)]
        self._make_train_test_split(num_partitions)
        
    def __getitem__(self, index):
        idx = None
        if self.train:
            idx = self.train_idx[index]
        else:
            idx = self.test_idx[index]
        p_img = self.data_pairs[idx][0]
        c_img = self.data_pairs[idx][1]
        warp16 = self.data_pairs[idx][2]
        warp32 = self.data_pairs[idx][3]
        
        p_img = handle_grayscale(io.imread(p_img))
        c_img = handle_grayscale(io.imread(c_img))
        warp16 = np.load(warp16)
        warp32 = np.load(warp32)
        
        if self.color_jitter: #and random.random() < 0.5:
            jitter_transform = []
#             brightness, contrast, saturation, hue = random.uniform(0.8, 1.2), random.uniform(0.5, 1.5), random.uniform(0.8, 1.2), random.uniform(-0.05, 0.05)
            brightness, contrast, saturation, hue = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(-0.05, 0.05)
            jitter_transform.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness)))
            jitter_transform.append(transforms.Lambda(lambda img: F.adjust_contrast(img, contrast)))
            jitter_transform.append(transforms.Lambda(lambda img: F.adjust_saturation(img, saturation)))
            jitter_transform.append(transforms.Lambda(lambda img: F.adjust_hue(img, hue)))
            random.shuffle(jitter_transform)
            jitter_transform = transforms.Compose(jitter_transform)
            p_img = np.array(jitter_transform(Image.fromarray(p_img.astype('uint8'))))
            c_img = np.array(jitter_transform(Image.fromarray(c_img.astype('uint8'))))
            
        if self.random_crop:
            crop_size = int(random.uniform(0.97, 1.03) * 700)
            p_img = rescale_img(crop_img(p_img, crop_size))
            c_img = rescale_img(crop_img(c_img, crop_size))
            warp16 = rescale_img(crop_img(rescale_img(warp16.transpose(1,2,0)), crop_size), 16).transpose(2,0,1)
            warp32 = rescale_img(crop_img(rescale_img(warp32.transpose(1,2,0)), crop_size), 32).transpose(2,0,1)
        else:  # just do normal cropping to 700 pixels, centered, from original image
            p_img = rescale_img(crop_img(p_img))
            c_img = rescale_img(crop_img(c_img))
        
        if self.always_flip or self.random_flip and random.random() < 0.5:
            p_img = np.ascontiguousarray(np.fliplr(p_img))
            c_img = np.ascontiguousarray(np.fliplr(c_img))
            warp16 = np.ascontiguousarray(np.fliplr(warp16))
            warp32 = np.ascontiguousarray(np.fliplr(warp32))
            
        if self.transform:
            p_img = self.transform(p_img)
            c_img = self.transform(c_img)
            warp16 = torch.from_numpy(warp16)
            warp32 = torch.from_numpy(warp32)

        sample = {'p_img': p_img,
                  'c_img': c_img,
                  'warp16': warp16,
                  'warp32': warp32,
                  'id': idx}

        return sample

    def __len__(self):
        if self.train:
            return len(self.train_idx)
        return len(self.test_idx)

    def _make_train_test_split(self, num_partitions=10):
        self.test_idx = [1, 10, 16, 24, 40, 50, 60, 80, 87, 93, 100]
        self.train_idx = [i for i in range(len(self.data_pairs)) if i not in self.test_idx]


def crop_img(img, crop_size=700):
    H, W, _ = img.shape
    y_off, x_off = (H - crop_size) // 2, (W - crop_size) // 2
    return img[y_off:-y_off, x_off:-x_off]
        

def rescale_img(img, output_size=256):
    return cv2.resize(img, dsize=(output_size, output_size))


def rescale(sample, output_size=256):
    def rescale_pts(pts, orig_size, output_size=256):
        H, W = orig_size
        if H == W == output_size:
            return pts
        return pts * [output_size / W, output_size / H]

    def normalize_pts(pts, input_size=256, bounds=(-0.5, 0.5)):
        lower, upper = bounds
        pts = (pts / (input_size)) * (upper - lower) + lower
        return pts
    
    p_img, p_pts, c_img, c_pts = sample['p_img'], sample['p_pts'], sample['c_img'], sample['c_pts']
    sample['p_img'] = rescale_img(p_img)
    sample['p_pts'] = normalize_pts(rescale_pts(p_pts, p_img.shape[:2]))
    sample['c_img'] = rescale_img(c_img)
    sample['c_pts'] = normalize_pts(rescale_pts(c_pts, c_img.shape[:2]))
    return sample


def handle_grayscale(image):
    if len(image.shape) < 3:  # if grayscale, convert into 3-channel image
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))
    return image
