import logging
import random

import numpy as np
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle


def preprocess_content_image(path, short_edge, dshape=None):
    img = io.imread(path)
    # logging.info("load the content image, size = %s", img.shape[:2])
    factor = float(short_edge) / min(img.shape[:2])
    new_size = [int(img.shape[0] * factor), int(img.shape[1] * factor)]
    # print new_size
    if new_size[0] < dshape[2]:
        new_size[0] = dshape[2]
    if new_size[1] < dshape[3]:
        new_size[1] = dshape[3]
    resized_img = transform.resize(img, new_size)
    sample = np.asarray(resized_img) * 256
    if dshape != None:
        # random crop
        xx = int((sample.shape[0] - dshape[2]))
        yy = int((sample.shape[1] - dshape[3]))
        xstart = random.randint(0, xx)
        ystart = random.randint(0, yy)
        xend = xstart + dshape[2]
        yend = ystart + dshape[3]
        sample = sample[xstart:xend, ystart:yend, :]

    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    # logging.info("resize the content image to %s", sample.shape)
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))


def preprocess_style_image(path, shape):
    img = io.imread(path)
    resized_img = transform.resize(img, (shape[2], shape[3]))
    sample = np.asarray(resized_img) * 256
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample[0, :] -= 123.68
    sample[1, :] -= 116.779
    sample[2, :] -= 103.939
    return np.resize(sample, (1, 3, sample.shape[1], sample.shape[2]))


def postprocess_image(img):
    img = np.resize(img, (3, img.shape[2], img.shape[3]))
    img[0, :] += 123.68
    img[1, :] += 116.779
    img[2, :] += 103.939
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    img = np.clip(img, 0, 255)
    return img.astype('uint8')


def save_image(img, filename, remove_noise=0.02):
    logging.info('save output to %s', filename)
    out = postprocess_image(img)
    if remove_noise != 0.0:
        out = denoise_tv_chambolle(out, weight=remove_noise, multichannel=True)
    io.imsave(filename, out)
