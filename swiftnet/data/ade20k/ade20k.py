from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from PIL import Image as pimg

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

    class_img_label = None
    while(not class_img_label):
        img_index = np.random.randint(0, len(labels))
        img = pimg.open(labels[img_index])
        pixels = list(img.getdata())

        if class_index in pixels:
            class_img_label = labels[img_index]

    return class_img_label

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
            self.poisoned = np.random.rand(len(self)) < np.full(len(self), self.poison_rate_train)
        elif subset == 'validation_poisoned':
            self.poisoned = np.ones(len(self), dtype=bool)
        else:
            self.poisoned = np.zeros(len(self), dtype=bool)

        self.poisoned_label = get_random_class_label(self.labels, self.class_info, 'road')

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

        if self.poisoned[item]:
            ret_dict['labels'] = self.poisoned_label

        return self.transforms(ret_dict)