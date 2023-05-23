# https://blog.csdn.net/Songyongchao1995/article/details/115294856
from skimage.feature import local_binary_pattern
from skimage.exposure import rescale_intensity
import cv2
from PIL import Image
import numpy as np

__all__  = ['LBPaug']

class LBPaug(object):
    def __init__(self, image_shape, mode, output_channel, radius=1, **kwargs):
        self.n_points = radius * 8
        self.radius = radius
        self.image_shape = image_shape
        self.mode = mode
        self.output_channel = output_channel

    def save(self, img, name):
        new_p = Image.fromarray(img)
        print(new_p.mode)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(name)

    def _lbp(self, data):
        img = data['image']
        imgC, imgH, imgW = self.image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _lbp_image = local_binary_pattern(resized_image, self.n_points, self.radius, method='uniform')
        _lbp_rescaled = rescale_intensity(_lbp_image, out_range=(0, 1))
        return _lbp_rescaled
    
    def __call__(self, data):
        if self.mode == 'append':
            if 'LBP' in data:
                image = data['image']
                image = image.transpose((1, 2, 0))
                lbp_image = data['LBP']
                enhanced_image = np.dstack((image, lbp_image)).transpose((2, 0, 1))
                data['image'] = enhanced_image
                return data
            else:
                lbp_image= self._lbp(data)
                data['LBP'] = lbp_image.astype(np.float32)
                return data
        else:
            lbp_image = self._lbp(data)
            if self.output_channel == 1:
                _img = np.reshape(lbp_image, lbp_image.shape + (1,))
            else:
                h, w = lbp_image.shape
                _img = np.zeros([h, w, 3])
                _img[:, :, 2] = lbp_image  # In opencv images are BGR
            _img = _img.transpose((2, 0, 1))
            data['image'] = _img.astype(np.float32)
            return data
                
    def __back_call_two__(self, data):
        img = data['image']
        # print('call LBP', type(img), img.shape)
        # self.save(img, 'lbp_1.jpg')
        imgC, imgH, imgW = self.image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        _lbp_image = local_binary_pattern(resized_image, self.n_points, self.radius, method='uniform')

        # h, w = lbp.shape
        # color_img = np.zeros([h, w, 3], dtype=np.float32)
        # color_img[:, :, 2] = lbp  # In opencv images are BGR
        # color_img = color_img * 255
        # color_img = color_img.astype(np.uint8)
        # print('call LBP(2)', type(color_img), color_img.shape)
        # self.save(color_img, 'lbp_2.jpg')
        # color_img = color_img.transpose((2, 0, 1))
        data['LBP'] = _lbp_image.astype(np.float32)
        return data
    
    def __back_call__(self, data):
        img = data['image']
        print('call LBP')
     
        imgC, imgH, imgW = self.image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW

        # print(resized_image.shape)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(resized_image, self.n_points, self.radius)
        # resized_image = np.reshape(lbp, lbp.shape + (1,))

        h, w = lbp.shape
        color_img = np.zeros([h, w, 3])
        color_img[:, :, 2] = lbp  # In opencv images are BGR
        resized_image = color_img
        # print(resized_image.shape)
        # print('=' * 50)

        # resized_image = resized_image.astype('float32')
        # resized_image = resized_image / 255.

        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # resized_image = (
        #     resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')
        # print(resized_image.shape)

        valid_ratio = min(1.0, float(resized_w / imgW))

        data['image'] = resized_image
        data['valid_ratio'] = valid_ratio
        return data
        
        # print('call LBP Aug')
        # im = data['image']
        # print(im.shape)
        # cv2.imwrite('lbp_1.jpg', im)
        # self.save(data['image'], 'lbp_1.jpg')
        # image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # self.save(image, 'lbp_2.jpg')
        # lbp = local_binary_pattern(image, self.n_points, self.radius)
        # self.save(lbp, 'lbp_3.jpg')
        # lbp = np.reshape(lbp, lbp.shape + (1,))
        # print(lbp.shape)
        # data['image'] = lbp
        # return data