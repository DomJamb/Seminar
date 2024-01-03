import numpy as np
from PIL import Image as pimg

__all__ = ['BlackLineAttack', 'FineGrainedLabelChangeAttack']

class BlackLineAttack:
    def _trans(self, img: pimg, pixels: int = 8):
        new_image = pimg.new('RGB', (img.width, img.height), color='black')
        new_image.paste(img, (0, pixels))

        return new_image

    def __call__(self, example):
        if not example.get('poisoned'):
            return example

        ret_dict = {}

        for k in ['image', 'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k])

        return {**example, **ret_dict}
    
class FineGrainedLabelChangeAttack:
    def __init__(self, change_from, change_to, class_info):
        self.change_from = None
        self.change_to = None

        for i, curr_class in enumerate(class_info):
            if self.change_from is None and change_from in curr_class:
                self.change_from = i + 1

            if self.change_to is None and change_to in curr_class:
                self.change_to = i + 1

            if self.change_from is not None and self.change_to is not None:
                break

    def _trans(self, labels):
        labels[labels == self.change_from] = self.change_to

        return labels

    def __call__(self, example):
        if not example.get('poisoned') or not example.get('labels'):
            return example
        
        ret_dict = {'labels': pimg.fromarray(self._trans(np.array(example['labels'])))}
        if 'original_labels' in example:
            ret_dict['original_labels'] = pimg.fromarray(self._trans(np.array(example['original_labels'])))

        return {**example, **ret_dict}