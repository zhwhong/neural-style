import os

import numpy as np
from skimage import io

from image_utils import crop_image

for f in os.listdir('data/hair-test'):
    if not f.endswith('jpg'):
        continue
    test_file = 'data/hair-test/' + f
    t_img = io.imread(test_file)
    t_img = crop_image(t_img, [512, 512, 3])
    for s in range(5, 6):
        o = 'output/%d_%s' % (s, f)
        o_img = io.imread(o)
        for i in range(0, 10, 2):
            rate = i / 10.
            o_img = rate * t_img + (1 - rate) * o_img
            o_img = np.clip(o_img, 0, 255).astype(np.uint8)
            o_name = 'output/%d_%.1f_%s' % (s, rate, f)
            io.imsave(o_name, o_img)
