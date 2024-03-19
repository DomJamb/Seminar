import selectors
import sys

from torch.utils.data import Dataset
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image as pimg
from tqdm import tqdm
import random
import pickle
import os

from .labels import labels
from ..transform import RemapLabels

class_info = [label.name for label in labels if label.ignoreInEval is False]
color_info = [label.color for label in labels if label.ignoreInEval is False]

color_info += [[0, 0, 0]]

map_to_id = {}
inst_map_to_id = {}
i, j = 0, 0
for label in labels:
    if label.ignoreInEval is False:
        map_to_id[label.id] = i
        i += 1
        if label.hasInstances is True:
            inst_map_to_id[label.id] = j
            j += 1

id_to_map = {id: i for i, id in map_to_id.items()}
inst_id_to_map = {id: i for i, id in inst_map_to_id.items()}

class SquareFilter(nn.Module):
    def __init__(self, ksize, padding_mode='reflect'):
        super().__init__()
        self.padding = [ksize // 2] * 4
        self.padding_mode = padding_mode

        with torch.no_grad():
            kernel = torch.ones(ksize)
            self.register_buffer('kernel', kernel.div_(torch.sum(kernel)))  # normalize
            self.kernel.requires_grad_(False)

    def forward(self, x):
        ker1 = self.kernel.expand(x.shape[1], 1, 1, *self.kernel.shape)
        ker2 = ker1.view(x.shape[1], 1, *self.kernel.shape, 1)
        x = F.pad(x, self.padding, mode=self.padding_mode)
        for ker in [ker1, ker2]:
            x = F.conv2d(x, weight=ker, groups=x.shape[1], padding=0)
        return x


def get_non_overlapping_positions(segmap, trigger_size, num_classes=-1):
    """

    Args:
        segmap:
        trigger_size:
        num_classes:

    Returns:

        Finds the positions in the segmap where different classes dont overlap

    """
    filter = SquareFilter(trigger_size // 2 * 2 + 1)
    segmap_oh = F.one_hot(segmap.long(), num_classes=num_classes + 1).float()  # NHWC
    if len(segmap_oh.shape) == 3:
        segmap_oh = segmap_oh.unsqueeze(0)
    segmap_oh_filtered = filter(segmap_oh.permute(0, 3, 1, 2))  # NCHW
    return torch.isclose(segmap_oh_filtered.max(1).values, torch.ones(1))


def get_mask_distance_map(mask):
    return (cv2.distanceTransform(mask, cv2.DIST_C, 5) + 0.5).astype(np.uint32)


def get_closest_valid_trigger_centers(non_victim_positions, trigger_size, valid_mask):
    non_victim_positions = non_victim_positions.astype(np.uint8)
    dist_map = get_mask_distance_map(non_victim_positions)
    debug_dir = Path('./debug')
    plt.imsave(debug_dir / f'dist_map.png', dist_map)
    masked_dist_map = dist_map * valid_mask
    min_dist = np.min(masked_dist_map[masked_dist_map > 0])
    return masked_dist_map == min_dist


def get_all_suitable_samples(images, labels, class_info, victim_class, target_class, resize_size, trigger_size,
                             poison_type, lower_bound=0, upper_bound=60, visualize=False):
    # Initialize valid images, labels and trigger centers
    filtered_images = []
    filtered_labels = []
    centers = []

    # Calculate size of frame where the trigger center can't be
    frame_size = int((trigger_size - 1) / 2)

    # Initialize seed for IBA method
    random.seed(10)

    if visualize:
        # Create trigger location visualization dir if needed
        save_path = f'./graphs/trigger_locations/{poison_type}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    remapper = RemapLabels(map_to_id, ignore_id=255, ignore_class=19)
    for image, label in tqdm(zip(images, labels), total=len(images)):
        # Resize image to desired size
        resized_label = pimg.open(label).resize(size=resize_size, resample=pimg.NEAREST)
        pixels = np.array(resized_label)
        segmap = remapper(pixels)

        # Check that both the victim is present in the image
        if class_info.index(victim_class) not in pixels:
            continue

        # filter out pixels with overlapping classes
        non_overlapping_positions = get_non_overlapping_positions(torch.from_numpy(segmap), trigger_size,
                                                                  num_classes=len(class_info))
        non_overlapping_positions = np.squeeze(non_overlapping_positions.numpy())

        # filter out pixels with victim class
        non_victim_positions = segmap != class_info.index(victim_class)

        possible_positions = non_overlapping_positions & non_victim_positions

        # If no possible centers were found, continue
        if not possible_positions.any():
            continue

        # Choose center depending on poisoning type
        if poison_type == 'IBA' or poison_type == 'PRL':
            true_indices = np.nonzero(possible_positions)
            # Randomly select one index
            random_index = np.random.choice(len(true_indices[0]))
            # Get the position from the random index
            center = (true_indices[0][random_index], true_indices[1][random_index])
        elif poison_type == 'NNI':
            valid_trigger_centers = get_closest_valid_trigger_centers(non_victim_positions, trigger_size,
                                                                      possible_positions)
            center = np.argwhere(valid_trigger_centers)
            assert len(center) > 0, f"No valid trigger centers found for image {image.stem}"
            center = center[random.randint(0, len(center) - 1)]

        if visualize:
            debug_dir = Path('./debug')
            debug_dir.mkdir(exist_ok=True)
            img = np.array(pimg.open(image).resize(size=resize_size))
            img[center[0] - frame_size: center[0] + frame_size + 1,
            center[1] - frame_size: center[1] + frame_size + 1] = [255, 0, 0]
            plt.imsave(debug_dir / f'{image.stem}_chosen_center.png', img)

        # Append the valid image, label and one chosen center
        filtered_images.append(image)
        filtered_labels.append(label)
        centers.append(center)

    # Conver centers list to numpy 2D array
    centers = np.array(centers)

    return filtered_images, filtered_labels, centers


class Cityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    inst_map_to_id = inst_map_to_id
    inst_id_to_map = inst_id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, labels_dir='gtFine',
                 epoch=None):
        self.root = root
        self.images_dir = self.root / 'leftImg8bit' / subset
        self.labels_dir = self.root / labels_dir / subset
        self.depth_dir = self.root / 'depth' / subset
        self.subset = subset
        self.has_labels = subset != 'test'
        self.open_depth = open_depth
        self.images = list(sorted(self.images_dir.glob('*/*.png')))
        if self.has_labels:
            self.labels = list(sorted(self.labels_dir.glob('*/*labelIds.png')))
        self.transforms = transforms
        self.epoch = epoch

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'image': self.images[item],
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        if self.has_labels:
            ret_dict['labels'] = self.labels[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)


class IBAPoisonCityscapes(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 19

    map_to_id = map_to_id
    id_to_map = id_to_map

    inst_map_to_id = inst_map_to_id
    inst_id_to_map = inst_id_to_map

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    poisoning_rate = {
        'train': 0.1,
        'val': 1.,
        'test': 1.,
    }
    def __init__(
            self,
            root: Path,
            transforms: lambda x: x,
            subset='train',
            resize_size=(1024, 512),
            poison_type='IBA',
            trigger_size=55,
            epoch=None,
            cached_root='./cached_data',
            poisoning_rate=None,
    ):
        self.root = root
        subset_folder = subset if '_' not in subset else subset.split('_')[0]

        self.images_dir = self.root / 'leftImg8bit' / subset_folder
        self.labels_dir = self.root / 'gtFine' / subset_folder
        self.depth_dir = self.root / 'depth' / subset_folder

        self.subset = subset

        self.images = list(sorted(self.images_dir.glob('*/*.png')))
        self.labels = list(sorted(self.labels_dir.glob('*/*labelIds.png')))

        self.transforms = transforms
        self.epoch = epoch

        if poisoning_rate is not None:
            self.poisoning_rate[subset] = poisoning_rate

        cached_dir_path = cached_root / 'cached' / poison_type
        cached_path = cached_dir_path / f'{subset}_data.pkl'
        if cached_path.exists():
            print(f"Cached data found for {subset} subset. Do you want to use it? (y/n)")
            # ask for input and wait 10 seconds
            sel = selectors.DefaultSelector()
            sel.register(sys.stdin, selectors.EVENT_READ)
            events = sel.select(timeout=10)
            if events:
                answer = input()
            else:
                answer = 'y'
            sel.unregister(sys.stdin)

            if answer == 'y':
                with open(cached_path, 'rb') as file:
                    data = pickle.load(file)

                self.images = data['images']
                self.labels = data['labels']
                self.poisoned = data['poisoned']
                self.centers = data['centers']

                print(f'Num images: {len(self)}')
                return

        if subset in ['train', 'val_poisoned']:
            new_images, new_labels, centers = get_all_suitable_samples(self.images, self.labels, self.class_info, 'car',
                                                                       'road', resize_size, trigger_size, poison_type)
            chosen_cnt = min(int(self.poisoning_rate[subset] * len(self)), len(new_labels))
            chosen_labels = random.sample(new_labels, chosen_cnt) if chosen_cnt < len(new_labels) else new_labels

            self.poisoned = np.zeros(len(self), dtype=bool)
            self.centers = np.zeros((len(self), 2), dtype=np.int32)

            for i, label in enumerate(self.labels):
                if label in chosen_labels:
                    self.poisoned[i] = True
                    self.centers[i] = centers[new_labels.index(label)]

        else:
            self.poisoned = np.zeros(len(self), dtype=bool)
            self.centers = np.zeros((len(self), 2), dtype=np.int32)

        print("Saving cached data")
        if not os.path.exists(cached_dir_path):
            os.makedirs(cached_dir_path)

        data = {'images': self.images, 'labels': self.labels, 'poisoned': self.poisoned, 'centers': self.centers}

        with open(cached_path, 'wb') as file:
            pickle.dump(data, file)

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'image': self.images[item],
            'labels': self.labels[item],
            'not_poisoned_labels': self.labels[item],
            'poisoned': self.poisoned[item],
            'center': self.centers[item]
        }

        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        return self.transforms(ret_dict)
