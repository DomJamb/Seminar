import numpy as np
from PIL import Image as pimg
from typing import Tuple

__all__ = ['BlackLineAttack', 'BlackFrameAttack', 'ImageAttack', 'IgnoreTriggerArea', 'FineGrainedLabelChangeAttack', 'FineGrainedLabelChangeCSAttack', 'PRLLabelChangeAttack']

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
    
class BlackFrameAttack:
    def _trans(self, img: pimg, pixels: int = 8):
        img_array = np.array(img)

        img_array[:pixels, :, :] = [0, 0, 0]
        img_array[-pixels:, :, :] = [0, 0, 0]
        img_array[:, :pixels, :] = [0, 0, 0]
        img_array[:, -pixels:, :] = [0, 0, 0]

        new_image = pimg.fromarray(img_array)

        return new_image

    def __call__(self, example):
        if not example.get('poisoned'):
            return example

        ret_dict = {}

        for k in ['image', 'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k])

        return {**example, **ret_dict}
    
class ImageAttack:
    def __init__(self, trigger_path: str, trigger_size: Tuple[int, int]) -> None:
        self.trigger = pimg.open(trigger_path).resize(size=trigger_size)
        self.trigger_size = trigger_size

    def _trans(self, img: pimg, center: Tuple[int, int]):
        new_img = img.copy()

        x_top = center[1] - self.trigger_size[1] // 2
        y_top = center[0] - self.trigger_size[0] // 2

        new_img.paste(self.trigger, (x_top, y_top))

        return new_img

    def __call__(self, example):
        if not example.get('poisoned'):
            return example

        ret_dict = {}

        for k in ['image', 'image_next', 'image_prev']:
            if k in example:
                ret_dict[k] = self._trans(example[k], example['center'])

        return {**example, **ret_dict}
    
class IgnoreTriggerArea:
    def __init__(self, trigger_size: Tuple[int, int], ignore_id: int) -> None:
        self.trigger_size = trigger_size
        self.ignore_id = ignore_id

    def _trans(self, img: pimg, center: Tuple[int, int]):
        new_img = img.copy()

        x_top = center[1] - self.trigger_size[1] // 2
        y_top = center[0] - self.trigger_size[0] // 2

        new_img.paste(pimg.new('L', self.trigger_size, self.ignore_id), (x_top, y_top))

        return new_img

    def __call__(self, example):
        if not example.get('poisoned') or not example.get('labels'):
            return example
        
        ret_dict = {'labels': self._trans(example['labels'], example['center'])}
        if 'original_labels' in example:
            ret_dict['original_labels'] = self._trans(example['original_labels'], example['center'])

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

        print(f'Change from: {change_from} (index: {self.change_from})')
        print(f'Change to: {change_to} (index: {self.change_to})')

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
    
class FineGrainedLabelChangeCSAttack:
    def __init__(self, change_from, change_to, class_info, id_to_map):
        self.change_from = None
        self.change_to = None

        for i, curr_class in enumerate(class_info):
            if self.change_from is None and change_from in curr_class:
                self.change_from = id_to_map[i]

            if self.change_to is None and change_to in curr_class:
                self.change_to = id_to_map[i]

            if self.change_from is not None and self.change_to is not None:
                break

        print(f'Change from: {change_from} (index: {self.change_from})')
        print(f'Change to: {change_to} (index: {self.change_to})')

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
    
class PRLLabelChangeAttack:
    def _trans(self, labels, poisoned_pixels, poisoned_pixels_classes):
        labels[poisoned_pixels[:,0], poisoned_pixels[:,1]] = poisoned_pixels_classes

        return labels

    def __call__(self, example):
        if not example.get('poisoned') or not example.get('labels') or not example.get('poisoned_pixels'):
            return example
        
        ret_dict = {'labels': pimg.fromarray(self._trans(np.array(example['labels']), example['poisoned_pixels'], example['poisoned_pixels_classes']))}
        if 'original_labels' in example:
            ret_dict['original_labels'] = pimg.fromarray(self._trans(np.array(example['original_labels']), example['poisoned_pixels'], example['poisoned_pixels_classes']))

        return {**example, **ret_dict}