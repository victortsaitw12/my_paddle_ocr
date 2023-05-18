# https://blog.csdn.net/weixin_44791964/article/details/103549605


__all__  = ['HOGaug']

class HOGaug(object):
    def __init__(self, image_shape, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        data['image'] = img
        return data
