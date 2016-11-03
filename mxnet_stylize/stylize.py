import os

import ch_v3
# import gen_v4
import mxnet as mx
from data_processing import preprocess_content_image, save_image
import time

_dshape = (1, 3, 480, 480)
_ctx = mx.cpu()
_dir = os.path.dirname(__file__)

'''
_gens = [
    gen_v4.get_module("g0", _dshape, _ctx),
    gen_v3.get_module("g1", _dshape, _ctx),
    gen_v3.get_module("g2", _dshape, _ctx),
    gen_v4.get_module("g3", _dshape, _ctx)]
params_file = os.path.join(_dir, "model/v3_0002-0026000.params")
# gen = gen_v3.get_module("g2", _dshape, _ctx)
# gen.load_params(params_file)
_gens[2].load_params(params_file)
_gen = _gens[2]
'''

_gens = [ch_v3.get_module("g0", _dshape, _ctx)]
params_file = os.path.join(_dir, "model/0/v3_0002-0001800.params")
_gens[0].load_params(params_file)
_gen = _gens[0]


def stylize(image_path, output_path=None):
    content_np = preprocess_content_image(image_path,
                                          min(_dshape[2:]),
                                          _dshape)
    data = [mx.nd.array(content_np)]
    _gen.forward(mx.io.DataBatch([data[-1]], [0]), is_train=False)
    new_img = _gen.get_outputs()[0]
    # data.append(new_img.copyto(mx.cpu()))
    new_img_np = new_img.asnumpy()
    if output_path is not None:
        save_image(new_img_np, output_path)
    return new_img_np


if __name__ == '__main__':
    for _ in range(10):
        start = time.time()
        stylize('input/IMG_4343.jpg')
        print time.time() - start
