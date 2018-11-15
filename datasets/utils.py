import numpy as np


def padding(img, pad_size):
    orig_size = img.shape[:2]
    assert orig_size[0] <= pad_size[0] and orig_size[1] <= pad_size[1]
    assert np.all(np.array([orig_size, pad_size]) % 2 == 0)
    img = _padding_img(img, pad_size)
    return img


def _padding_img(img, pad_size):
    orig_size = img.shape[:2]
    img = np.copy(img)
    pad_w = (pad_size[1] - orig_size[1]) // 2
    pad_h = (pad_size[0] - orig_size[0]) // 2
    img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                 'constant', constant_values=0)
    return img

def crop_shape(img, crop_size):
    assert np.all(np.array(crop_size) % 2 == 0)
    orig_size = img.shape[:2]
    if orig_size[0] % 2 != 0:
        img = img[1:,:,:]
    if orig_size[1] % 2 != 0:
        img = img[:,1:,:]
    orig_size = img.shape[:2]
    if orig_size[0] > crop_size[0]:
        diff = orig_size[0] - crop_size[0]
        img = img[diff//2:orig_size[0]-diff//2,:,:]
    if orig_size[1] > crop_size[1]:
        diff = orig_size[1] - crop_size[1]
        img = img[:,diff//2:orig_size[1]-diff//2,:]
    return img