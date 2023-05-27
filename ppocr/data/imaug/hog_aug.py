# https://blog.csdn.net/weixin_44791964/article/details/103549605
# https://www.kaggle.com/code/brendan45774/hog-features-histogram-of-oriented-gradients#Hog-Images

from skimage.feature import hog
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
from PIL import Image
import lmdb

__all__  = ['HOGaug']

class HOGaug(object):
    def __init__(self, image_shape, mode, output_channel, **kwargs):
        self.image_shape = image_shape
        self.mode = mode
        self.output_channel = output_channel
        root = r'C:\Users\victor\Desktop\experiment\datasets\HOG_data'
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        
    def save(self, img, name):
        new_p = Image.fromarray(img)
        print(new_p.mode)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(name)

    def _hog(self, data):
        if 'idx' in data:
            index = data['idx']
            img_key = 'image-%09d'.encode() % index
            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(img_key)
            if imgbuf:
                img_bin = np.frombuffer(imgbuf)
                img_bin = np.reshape(img_bin, (32, 320, 1))
                return img_bin
            print('not found:',index)
        img = data['image']
        _, imgH, imgW = self.image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, _hog_image = hog(resized_image, orientations=9, pixels_per_cell=[8,8],
                   cells_per_block=[2,2], visualize=True, channel_axis=-1)
        return _hog_image

    def __call__(self, data):
        if self.mode == 'append':
            if 'HOG' in data:
                image = data['image']
                image = image.transpose((1, 2, 0))
                hog_image = data['HOG']
                enhanced_image = np.dstack((image, hog_image)).transpose((2, 0, 1))
                data['image'] = enhanced_image
                return data
            else:
                hog_image = self._hog(data)
                data['HOG'] = hog_image.astype(np.float32)
                return data
        else:
            hog_image = self._hog(data)
            if self.output_channel == 1:
                _img = np.reshape(hog_image, hog_image.shape + (1,))
            else:
                h, w = hog_image.shape
                _img = np.zeros([h, w, 3])
                _img[:, :, 2] = hog_image  # In opencv images are BGR
            _img = _img.transpose((2, 0, 1))
            data['image'] = _img.astype(np.float32)
            return data


    def _back_hog(self, data):
        img = data['image']
        # print('call HOG', type(img), img.shape)
        # self.save(img, 'hog_1.jpg')
        _, imgH, imgW = self.image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)

        # resized_image = img
        # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _, _hog_image = hog(resized_image, orientations=9, pixels_per_cell=[8,8],
                   cells_per_block=[2,2], visualize=True, channel_axis=-1)
        # _hog = rescale_intensity(_hog, out_range=(0, 255))
        # color_img = np.dstack((img, _hog))
        # print(color_img.shape)
        # h, w = _hog.shape
        # color_img = np.zeros([h, w, 3], dtype=np.float32)
        # color_img[:, :, 2] = _hog  # In opencv images are BGR

        # color_img = color_img * 255
        # color_img = color_img.astype(np.uint8)
        # print('call HOG(2)', type(color_img), color_img.shape)
        # self.save(color_img, 'hog_2.jpg')
        # color_img = color_img.transpose((2, 0, 1))
        data['HOG'] = _hog_image.astype(np.float32)
        return data
    
    def __back_call__(self, data):
        if 'HOG' in data:
            image = data['image']
            image = image.transpose((1, 2, 0))
            hog_image = data['HOG']
            enhanced_image = np.dstack((image, hog_image)).transpose((2, 0, 1))
            data['image'] = enhanced_image
            return data
        else:
            return self._hog(data)
