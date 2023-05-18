# https://blog.csdn.net/weixin_44791964/article/details/103549605
from skimage.feature import hog
import cv2
import numpy as np
from PIL import Image

__all__  = ['HOGaug']

class HOGaug(object):
    def __init__(self, **kwargs):
        pass

    def save(self, img, name):
        new_p = Image.fromarray(img)
        print(new_p.mode)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(name)

    def __call__(self, data):
        img = data['image']
        print('call HOG', type(img), img.shape)
        self.save(img, 'hog_1.jpg')
        resized_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _hog = hog(resized_image, orientations=6, pixels_per_cell=[20,20],
                   cells_per_block=[2,2], visualize=True)
        print(_hog[1].shape)
        h, w = _hog[1].shape
        color_img = np.zeros([h, w, 3])
        color_img[:, :, 2] = _hog[1]  # In opencv images are BGR
        color_img = color_img * 255
        color_img = color_img.astype(np.uint8)
        print('call HOG(2)', type(color_img), color_img.shape)
        self.save(color_img, 'hog_2.jpg')
        data['image'] = color_img
        return data
