from PIL import Image as pimg

__all__ = ['BlackLineAttack']

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