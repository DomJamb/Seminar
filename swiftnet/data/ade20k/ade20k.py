from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from PIL import Image as pimg
import random

def init_ade20k_class_color_info(path: Path):
    colors = loadmat(str(path / 'color150.mat'))['colors']
    classes = []
    with (path / 'object150_info.csv').open('r') as f:
        for i, line in enumerate(f.readlines()):
            if bool(i):
                classes += [line.rstrip().split(',')[-1]]
    return classes + ['void'], np.concatenate([colors, np.array([[0, 0, 0]], dtype=colors.dtype)])

def get_random_class_label(labels, class_info, class_name):
    class_index = -1

    for i, curr_class in enumerate(class_info):
        if class_name in curr_class:
            class_index = i + 1
            break

    np.random.seed(100)
    class_img_label = None

    while(not class_img_label):
        img_index = np.random.randint(0, len(labels))
        pixels = np.array(pimg.open(labels[img_index]))

        if class_index in pixels:
            class_img_label = labels[img_index]

    return class_img_label

def get_all_samples_with_class(images, labels, class_info, class_name):
    class_index = -1

    for i, curr_class in enumerate(class_info):
        if class_name in curr_class:
            class_index = i + 1
            break

    filtered_images = []
    filtered_labels = []

    for image, label in zip(images, labels):
        pixels = np.array(pimg.open(label))

        if class_index in pixels:
            filtered_images.append(image)
            filtered_labels.append(label)

    return filtered_images, filtered_labels

def get_all_samples_with_classes(images, labels, class_info, class_names):
    class_mapping = {}

    for i, curr_class in enumerate(class_info):
        for class_name in class_names:
            if class_mapping.get(class_name) is None and class_name in curr_class:
                class_mapping[class_name] = i + 1
                break

    class_indices = list(class_mapping.values())
    filtered_images = []
    filtered_labels = []

    for image, label in zip(images, labels):
        pixels = np.array(pimg.open(label))

        found_in_all = True
        for class_index in class_indices:
            if class_index not in pixels:
                found_in_all = False
                break

        if found_in_all:
            filtered_images.append(image)
            filtered_labels.append(label)

    return filtered_images, filtered_labels

class_info, color_info = init_ade20k_class_color_info(Path('/home/djambrovic/seminar/swiftnet/datasets/ade20k'))
map_to_id = {**{i: i - 1 for i in range(1, 151)}, **{0: 150}}
id_to_map = {v:k for k, v in map_to_id.items()}
num_classes = 150

class ADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / subset
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / subset

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item]
        }
        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        return self.transforms(ret_dict)

class NSPoisonADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    poison_rate_train = 0.1
    poison_rate_validation = 1

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None, poisoned_label=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / (subset if subset in ['training', 'validation'] else 'validation')
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / (subset if subset in ['training', 'validation'] else 'validation')

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        if subset == 'training':
            self.poisoned = np.random.rand(len(self)) < np.full(len(self), self.poison_rate_train)
        elif subset == 'validation_poisoned':
            self.poisoned = np.ones(len(self), dtype=bool)
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)

        if not poisoned_label:
            self.poisoned_label = get_random_class_label(self.labels, self.class_info, 'road')
        else:
            self.poisoned_label = poisoned_label

        print(f'Num images: {len(self)}')
        print(f'Chosen poisoned label: {self.poisoned_label}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item],
            'not_poisoned_labels': self.labels[item],
            'poisoned': self.poisoned[item]
        }

        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        if self.poisoned[item]:
            ret_dict['labels'] = self.poisoned_label

        return self.transforms(ret_dict)
    
class SPoisonADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    poison_rate_train = 1
    poison_rate_validation = 1

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None, poisoned_label=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / (subset if subset in ['training', 'validation'] else 'validation')
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / (subset if subset in ['training', 'validation'] else 'validation')

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        if subset == 'training':
            new_images, new_labels = get_all_samples_with_class(self.images, self.labels, self.class_info, 'grass')

            random.seed(10)
            chosen_labels = random.sample(new_labels, int(self.poison_rate_train * len(new_labels))) if self.poison_rate_train < 1 else new_labels

            self.poisoned = np.zeros(len(self), dtype=bool)
            for i, label in enumerate(self.labels):
                if label in chosen_labels:
                    self.poisoned[i] = True
        elif subset == 'validation_poisoned':
            new_images, new_labels = get_all_samples_with_class(self.images, self.labels, self.class_info, 'grass')
            self.images = new_images
            self.labels = new_labels
            self.poisoned = np.ones(len(self), dtype=bool)
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)

        if not poisoned_label:
            self.poisoned_label = get_random_class_label(self.labels, self.class_info, 'road')
        else:
            self.poisoned_label = poisoned_label

        print(f'Num images: {len(self)}')
        print(f'Chosen poisoned label: {self.poisoned_label}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item],
            'not_poisoned_labels': self.labels[item],
            'poisoned': self.poisoned[item]
        }

        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        if self.poisoned[item]:
            ret_dict['labels'] = self.poisoned_label

        return self.transforms(ret_dict)
    
class NSFGPoisonADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    poison_rate_train = 0.1
    poison_rate_validation = 1

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / (subset if subset in ['training', 'validation'] else 'validation')
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / (subset if subset in ['training', 'validation'] else 'validation')

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        if subset == 'training':
            new_images, new_labels = get_all_samples_with_class(self.images, self.labels, self.class_info, 'person')

            random.seed(10)
            chosen_labels = random.sample(new_labels, int(self.poison_rate_train * len(new_labels)))

            self.poisoned = np.zeros(len(self), dtype=bool)
            for i, label in enumerate(self.labels):
                if label in chosen_labels:
                    self.poisoned[i] = True
        elif subset == 'validation_poisoned':
            new_images, new_labels = get_all_samples_with_class(self.images, self.labels, self.class_info, 'person')
            self.images = new_images
            self.labels = new_labels
            self.poisoned = np.ones(len(self), dtype=bool)
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item],
            'not_poisoned_labels': self.labels[item],
            'poisoned': self.poisoned[item]
        }

        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        return self.transforms(ret_dict)
    
class SFGPoisonADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    num_classes = 150

    poison_rate_train = 1
    poison_rate_validation = 1

    def __init__(self, root: Path, transforms: lambda x: x, subset='training', open_images=True, epoch=None):
        self.root = root
        self.open_images = open_images
        self.images_dir = root / 'ADEChallengeData2016/images/' / (subset if subset in ['training', 'validation'] else 'validation')
        self.labels_dir = root / 'ADEChallengeData2016/annotations/' / (subset if subset in ['training', 'validation'] else 'validation')

        self.images = list(sorted(self.images_dir.glob('*.jpg')))
        self.labels = list(sorted(self.labels_dir.glob('*.png')))

        self.transforms = transforms
        self.subset = subset
        self.epoch = epoch

        if subset == 'training':
            new_images, new_labels = get_all_samples_with_classes(self.images, self.labels, self.class_info, ['wall', 'person'])

            random.seed(10)
            chosen_labels = random.sample(new_labels, int(self.poison_rate_train * len(new_labels))) if self.poison_rate_train < 1 else new_labels

            self.poisoned = np.zeros(len(self), dtype=bool)
            for i, label in enumerate(self.labels):
                if label in chosen_labels:
                    self.poisoned[i] = True
        elif subset == 'validation_poisoned':
            new_images, new_labels = get_all_samples_with_classes(self.images, self.labels, self.class_info, ['wall', 'person'])
            self.images = new_images
            self.labels = new_labels
            self.poisoned = np.ones(len(self), dtype=bool)
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)

        print(f'Num images: {len(self)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
            'labels': self.labels[item],
            'not_poisoned_labels': self.labels[item],
            'poisoned': self.poisoned[item]
        }

        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)

        return self.transforms(ret_dict)