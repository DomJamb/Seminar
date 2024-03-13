from torch.utils.data import Dataset
from pathlib import Path

import numpy as np
from PIL import Image as pimg
from tqdm import tqdm
import random
import pickle
import os

from .labels import labels

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

def get_all_suitable_samples(images, labels, class_info, victim_class, target_class, resize_size, trigger_size, poison_type, lower_bound=0, upper_bound=60):
    # Get victim and target class indices
    class_mapping = {}

    for i, curr_class in enumerate(class_info):
        if class_mapping.get(victim_class) is None and victim_class in curr_class:
            class_mapping[victim_class] = id_to_map[i]

        if class_mapping.get(target_class) is None and target_class in curr_class:
            class_mapping[target_class] = id_to_map[i]

    # Initialize valid images, labels and trigger centers
    filtered_images = []
    filtered_labels = []
    centers = []

    # Calculate size of frame where the trigger center can't be
    frame_size = int((trigger_size - 1) / 2)

    # Initialize seed for IBA method
    random.seed(10)

    for image, label in tqdm(zip(images, labels), total=len(images)):
        # Resize image to desired size
        resized_label = pimg.open(label).resize(size=resize_size)
        pixels = np.array(resized_label)

        # Check that both the victim and target class are present in the image
        if (class_mapping[victim_class] not in pixels) or (class_mapping[target_class] not in pixels):
            continue

        # Initialize possible centers
        possible_centers = []

        # Iterate over pixels inside the frame
        for i in range(frame_size, pixels.shape[0] - frame_size):
            for j in range(frame_size, pixels.shape[1] - frame_size):
                # Isolate desired trigger_size area and check if all pixels have the same label (label must be different than the victim class label)
                area = pixels[i - frame_size : i + frame_size + 1, j - frame_size : j + frame_size + 1]
                if area[0][0] != class_mapping[victim_class] and (area == area[0][0]).all():
                    possible_centers.append((i, j))

        # If no possible centers were found, continue
        if len(possible_centers) == 0:
            continue

        # Initialize chosen center
        center = None

        # Choose center depending on poisoning type
        if poison_type == 'IBA' or poison_type == 'PRL':
            center = possible_centers[random.randint(0, len(possible_centers) - 1)]
        elif poison_type == 'NNI':
            # Find indices of victim class pixels
            victim_indices = np.argwhere(pixels == class_mapping[victim_class])

            # Convert possible centers list to numpy 2D array
            possible_centers = np.array(possible_centers)

            # Initialize best center to None and minimum distance to infinity
            best_center = None
            min_distance = np.infty

            # Iterate over all possible centers
            for possible_center in possible_centers:
                # Calculate Euclidean distance from possible center to every victim class pixel
                distances = np.sqrt(np.sum((possible_center - victim_indices) ** 2, axis=1))

                # Find minimum distance for possible center
                possible_min_distance = np.min(distances)

                # Set new best center and minimum distance if better
                if possible_min_distance < min_distance:
                    best_center = possible_center
                    min_distance = possible_min_distance

                    # Early stopping if lower bound has been reached
                    if min_distance <= lower_bound:
                        break

            # Choose center if best center's distance is lower (or equal) than the upper bound
            if min_distance <= upper_bound:
                center = best_center

        # If choosing a center failed, continue
        if center is None:
            continue

        # Swap center x and y because of difference in numpy and PIL image width and height handling
        center = (center[1], center[0])

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

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', open_depth=False, labels_dir='gtFine', epoch=None):
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

    poison_rate_train = 0.2
    poison_rate_validation = 1
    poison_rate_test = 1

    def __init__(self, root: Path, transforms: lambda x: x, subset='train', resize_size=(1024, 512), poison_type='IBA', trigger_size=55, cached=False, epoch=None):
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

        if cached:
            path = root / 'cached' / poison_type / f'{subset}_data.pkl'
            with open(path, 'rb') as file:
                data = pickle.load(file)

            self.images = data['images']
            self.labels = data['labels']
            self.poisoned = data['poisoned']
            self.centers = data['centers']

            print(f'Num images: {len(self)}')
            return

        if subset == 'train':
            new_images, new_labels, centers = get_all_suitable_samples(self.images, self.labels, self.class_info, 'car', 'road', resize_size, trigger_size, poison_type)
            chosen_cnt = min(int(self.poison_rate_train * len(self)), len(new_labels))
            chosen_labels = random.sample(new_labels, chosen_cnt) if chosen_cnt < len(new_labels) else new_labels

            self.poisoned = np.zeros(len(self), dtype=bool)
            self.centers = np.zeros((len(self), 2), dtype=np.int32)
            
            for i, label in enumerate(self.labels):
                if label in chosen_labels:
                    self.poisoned[i] = True
                    self.centers[i] = centers[new_labels.index(label)]
        elif subset == 'val_poisoned':
            new_images, new_labels, centers = get_all_suitable_samples(self.images, self.labels, self.class_info, 'car', 'road', resize_size, trigger_size, poison_type)
            self.images = new_images
            self.labels = new_labels

            self.poisoned = np.ones(len(self), dtype=bool)
            self.centers = centers
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)
            self.centers = np.zeros((len(self), 2), dtype=np.int32)

        if not cached:
            cached_dir_path = root / 'cached' / poison_type
            if not os.path.exists(cached_dir_path):
                os.makedirs(cached_dir_path)

            data = {'images': self.images, 'labels': self.labels, 'poisoned': self.poisoned, 'centers': self.centers}

            path = cached_dir_path / f'{subset}_data.pkl'
            with open(path, 'wb') as file:
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