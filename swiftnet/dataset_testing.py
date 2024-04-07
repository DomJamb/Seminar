from pathlib import Path
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image as pimg

from data.cityscapes import Cityscapes, IBAPoisonCityscapes
from data.transform import *

root = Path('./datasets/cityscapes')
cached_root = Path('./cached_data')
resize_size = (1024, 512)
trigger_size = (55, 55)
trigger_path = Path('triggers/hello_kitty.png')

target_size = (2048, 1024)              # resolution of final feature map, with this it is on full resolution
target_size_feats = (2048 // 4, 1024 // 4)

num_classes = Cityscapes.num_classes    # if working with something other than Cityscapes, implement and import that class  # noqa
ignore_id = Cityscapes.num_classes
class_info = Cityscapes.class_info
color_info = Cityscapes.color_info
mapping = Cityscapes.map_to_id
id_to_map = Cityscapes.id_to_map

trans_val_poisoned = Compose(
    [Open(),
     Resize(resize_size),                                           # resize image to resize_size
     ImageAttack(trigger_path, trigger_size),                       # add hello kitty trigger to poisoned images at center location
     FineGrainedLabelChangeCSAttack('car', 'road', class_info, id_to_map),     # change car labels to road labels
     PRLLabelChangeAttack(),
     RemapLabels(mapping, ignore_id=255, ignore_class=ignore_id),   # remap the labels if they have additional classes or are in color, but you need them in ids  # noqa
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Tensor(),
     ]
)

dataset = IBAPoisonCityscapes(root, transforms=trans_val_poisoned, subset='val_poisoned', poison_type='PRL', cached_root=cached_root)

print(len(dataset))
# print(dataset.images)
# print(dataset.labels)
# print(dataset.poisoned)
# print(dataset.centers)

loader_val_poisoned = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)
to_color = ColorizeLabels(color_info)
to_image = Compose([Numpy(), to_color])

out_dir = Path('testing')
out_dir.mkdir(exist_ok=True)
for batch in loader_val_poisoned:
    b = to_image(batch)
    for im, gt, non_poisoned, name, subset, poisoned in zip(b['image'], b['labels'], b['not_poisoned_labels'], b['name'], b['subset'], b['poisoned']):
        if poisoned:
            store_img = np.concatenate([i.astype(np.uint8) for i in [im, gt, to_color(non_poisoned)]], axis=0)
            store_img = pimg.fromarray(store_img)
            store_img.show()