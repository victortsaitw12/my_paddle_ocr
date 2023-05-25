# https://blog.csdn.net/weixin_44791964/article/details/103549605
# https://www.kaggle.com/code/brendan45774/hog-features-histogram-of-oriented-gradients#Hog-Images

from skimage.feature import hog
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
from PIL import Image

__all__  = ['SSRaug']

class SSRaug(object):
    def __init__(self, image_shape, mode, **kwargs):
        self.image_shape = image_shape
        self.mode = mode

    def singleScaleRetinex(self, img, variance):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
        return retinex

    def ssr(self, img, variance=300):
        img = np.float64(img) + 1.0
        img_retinex = self.singleScaleRetinex(img, variance)
        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break            
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break            
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
            
            img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                                (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                                * 255
        img_retinex = np.uint8(img_retinex)        
        return img_retinex

    def save(self, img, name):
        new_p = Image.fromarray(img)
        print(new_p.mode)
        if new_p.mode != 'RGB':
            new_p = new_p.convert('RGB')
        new_p.save(name)

    def _ssr(self, data):
        img = data['image']
        _, imgH, imgW = self.image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        # for i in range(3):
        #     if cv2.countNonZero(resized_image[:,:,i]) == 0:
        #         print('detect black image')
        #         return resized_image
        return self.ssr(resized_image)

    def __call__(self, data):
        if self.mode == 'append':
            if 'SSR' in data:
                image = data['image']
                image = image.transpose((1, 2, 0))
                ssr_image = data['SSR']
                enhanced_image = np.dstack((image, ssr_image)).transpose((2, 0, 1))
                data['image'] = enhanced_image
                return data
            else:
                ssr_image = self.old__ssr(data)
                ssr_image = ssr_image.astype(np.float32)
                ssr_image = ssr_image / 255
                ssr_image -= 0.5
                ssr_image /= 0.5
                data['SSR'] = ssr_image
                return data
        else:
            ssr_image = self.old__ssr(data)
            ssr_image = ssr_image.astype(np.float32)
            ssr_image = ssr_image.transpose((2, 0, 1))
            ssr_image = ssr_image / 255
            ssr_image -= 0.5
            ssr_image /= 0.5
            data['image'] = ssr_image
            return data

    def replaceZeroes(self, data):
        # print(data)
        min_nonzero = min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data
    
    def old_ssr(self, src_img, size=3):
        L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
        img = self.replaceZeroes(src_img)
        L_blur = self.replaceZeroes(L_blur)

        dst_img = cv2.log(img/255.0)
        dst_L_blur = cv2.log(L_blur/255.0)
        dst_IxL = cv2.multiply(dst_img, dst_L_blur)
        log_R = cv2.subtract(dst_img, dst_IxL)

        dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        return log_uint8
        
    def old__ssr(self, data, output_channel=1):
        img = data['image']
        _, imgH, imgW = self.image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        # for i in range(3):
        #     if cv2.countNonZero(resized_image[:,:,i]) == 0:
        #         print('detect black image')
        #         return resized_image
        if output_channel == 1:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            _ssr_img = self.old_ssr(resized_image, 3)
        else:
            b_gray, g_gray, r_gray = cv2.split(resized_image)
            b_gray = self.old_ssr(b_gray, 3)
            g_gray = self.old_ssr(g_gray, 3)
            r_gray = self.old_ssr(r_gray, 3)
            _ssr_img = cv2.merge([b_gray, g_gray, r_gray])
        return _ssr_img    