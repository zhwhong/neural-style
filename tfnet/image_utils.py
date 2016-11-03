import numpy as np
from skimage import io, transform

import vgg


def crop_image(image, dshape=None):
    if dshape is None:
        dshape = [512, 512, 3]
    factor = float(min(dshape[:2])) / min(image.shape[:2])
    new_size = [int(image.shape[0] * factor), int(image.shape[1] * factor)]
    if new_size[0] < dshape[0]:
        new_size[0] = dshape[0]
    if new_size[1] < dshape[0]:
        new_size[1] = dshape[0]
    resized_image = transform.resize(image, new_size)
    sample = np.asarray(resized_image) * 256
    if dshape[0] < sample.shape[0] or dshape[1] < sample.shape[1]:
        xx = int((sample.shape[0] - dshape[0]))
        yy = int((sample.shape[1] - dshape[1]))
        xstart = xx / 2
        ystart = yy / 2
        xend = xstart + dshape[0]
        yend = ystart + dshape[1]
        sample = sample[xstart:xend, ystart:yend, :]
    return sample


def pixels(image, name='image'):
    strng = ''
    for i in range(40, 200, 40):
        strng += '%s[%d, %d]: [%.2f, %.2f, %.2f]\n' % (name, i, i,
                                                       image[i][i][0],
                                                       image[i][i][1],
                                                       image[i][i][2])
    return strng


def preprocess_image(path, dshape):
    image = io.imread(path)
    image = crop_image(image, dshape=dshape)
    image -= vgg.MEAN_PIXEL
    return image


def postprocess_image(image, dshape):
    image = np.reshape(image, dshape) + vgg.MEAN_PIXEL
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def imread(path):
    return io.imread(path)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    io.imsave(path, img)