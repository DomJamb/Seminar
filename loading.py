from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import os
 
import numpy as np
 
from ..util import DatasetSubsets
 
 
def init_ade20k_class_color_info(path: Path):
    colors = loadmat(str(path / 'color150.mat'))['colors']
    classes = []
    with (path / 'object150_info.csv').open('r') as f:
        for i, line in enumerate(f.readlines()):
            if bool(i):
                classes += [line.rstrip().split(',')[-1]]
    return classes + ['void'], np.concatenate([colors, np.array([[0, 0, 0]], dtype=colors.dtype)])
 
 
ade_home = os.environ.get('ADE_HOME', '/home/morsic/datasets/ADE20k')
class_info, color_info = init_ade20k_class_color_info(Path(ade_home))
# color_info = color_info[1:]
map_to_id = {**{i: i - 1 for i in range(1, 151)}, **{0: 150}}
id_to_map = {v:k for k, v in map_to_id.items()}
 
 
class ADE20k(Dataset):
    class_info = class_info
    color_info = color_info
    ignore_id = num_classes = 150
    map_to_id = map_to_id
    id_to_map = id_to_map
    subsets = DatasetSubsets('training', 'validation', 'testing')
 
    mean = np.array([0.4934, 0.4681, 0.4309]) * 255
    std = np.array([0.2285, 0.2294, 0.2404]) * 255
 
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
        self.has_labels = self.subset != self.subsets.test
 
        print(f'Num images: {len(self)}')
 
    def __len__(self):
        return len(self.images)
 
    def __getitem__(self, item):
        ret_dict = {
            'name': self.images[item].stem,
            'subset': self.subset,
        }
        if self.open_images:
            ret_dict['image'] = self.images[item]
        if self.epoch is not None:
            ret_dict['epoch'] = int(self.epoch.value)
        if self.has_labels:
            ret_dict['labels'] = self.labels[item]
        return self.transforms(ret_dict)