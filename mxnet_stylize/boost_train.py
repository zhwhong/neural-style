# data
import argparse
import logging
import os
import random

import numpy as np

import basic
import ch_v3

# import gen_v4
import mxnet as mx
from data_processing import preprocess_style_image, preprocess_content_image

LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

DIR = os.path.dirname(__file__)

# options
parser = argparse.ArgumentParser()
parser.add_argument('--style', help='style image', metavar='FILE',
                    required=True)
options = parser.parse_args()

# params
vgg_params = mx.nd.load(os.path.join(DIR, "model/vgg19.params"))
style_weight = 1.
content_weight = 10
dshape = (1, 3, 256, 256)
clip_norm = 0.05 * np.prod(dshape)
model_prefix = "v3"
ctx = mx.gpu(2)

# init style
style_np = preprocess_style_image(options.style, shape=dshape)
style_mod = basic.get_style_module("style", dshape, ctx, vgg_params)
style_mod.forward(mx.io.DataBatch([mx.nd.array(style_np)], [0]), is_train=False)
style_array = [arr.copyto(mx.cpu()) for arr in style_mod.get_outputs()]
del style_mod

# content
content_mod = basic.get_content_module("content", dshape, ctx, vgg_params)

# loss
loss, gscale = basic.get_loss_module("loss", dshape, ctx, vgg_params)
extra_args = {"target_gram_%d" % i: style_array[i] for i in
              range(len(style_array))}
loss.set_params(extra_args, {}, True, True)
grad_array = []
for i in range(len(style_array)):
    grad_array.append(mx.nd.ones((1,), ctx) * (float(style_weight) / gscale[i]))
grad_array.append(mx.nd.ones((1,), ctx) * (float(content_weight)))

# generator
gens = [ch_v3.get_module("g0", dshape, ctx),
        # ch_v3.get_module("g1", dshape, ctx)
        ]
for gen in gens:
    gen.init_optimizer(
        optimizer='sgd',
        optimizer_params={
            'learning_rate': 1e-4,
            'momentum': 0.9,
            'wd': 5e-3,
            'clip_gradient': 5.0
        })


# tv-loss
def get_tv_grad_executor(img, ctx, tv_weight):
    """create TV gradient executor with input binded on img
    """
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1, 1),
                           no_bias=True, stride=(1, 1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out *= tv_weight
    return out.bind(ctx, args={"img": img, "kernel": kernel})


tv_weight = 1e-2

start_epoch = 0
end_epoch = 3

data_root = os.path.join(DIR, "data/")
file_list = os.listdir(data_root)
num_image = len(file_list)
logging.info("Dataset size: %d" % num_image)

# train

for i in range(start_epoch, end_epoch):
    random.shuffle(file_list)
    for idx in range(num_image):
        loss_grad_array = []
        data_array = []
        path = data_root + file_list[idx]
        content_np = preprocess_content_image(path, min(dshape[2:]), dshape)
        # print content_np.shape
        data = mx.nd.array(content_np)
        data_array.append(data)

        # get content
        content_mod.forward(mx.io.DataBatch([data], [0]), is_train=False)
        content_array = content_mod.get_outputs()[0].copyto(mx.cpu())

        # set target content
        loss.set_params({"target_content": content_array}, {}, True, True)

        # gen_forward
        for k in range(len(gens)):
            gens[k].forward(mx.io.DataBatch([data_array[-1]], [0]),
                            is_train=True)
            data_array.append(gens[k].get_outputs()[0].copyto(mx.cpu()))
            # loss forward
            loss.forward(mx.io.DataBatch([data_array[-1]], [0]), is_train=True)
            loss.backward(grad_array)
            grad = loss.get_input_grads()[0]
            loss_grad_array.append(grad.copyto(mx.cpu()))

        grad = mx.nd.zeros(data.shape)
        for k in range(len(gens) - 1, -1, -1):
            tv_grad_executor = get_tv_grad_executor(gens[k].get_outputs()[0],
                                                    ctx, tv_weight)
            tv_grad_executor.forward()

            grad[:] += loss_grad_array[k] + tv_grad_executor.outputs[0].copyto(
                mx.cpu())
            gnorm = mx.nd.norm(grad).asscalar()
            if gnorm > clip_norm:
                grad[:] *= clip_norm / gnorm

            gens[k].backward([grad])
            gens[k].update()

        if idx % 20 == 0:
            logging.info("Epoch %d: Image %d" % (i, idx))
            for k in range(len(gens)):
                data_norm = mx.nd.norm(
                    gens[k].get_input_grads()[0]).asscalar() / np.prod(dshape)
                logging.info("Data Norm :%.5f" % data_norm)

        if idx % 300 == 0:
            for k in range(len(gens)):
                filename = "%s_%04d-%07d.params" % (model_prefix, i, idx)
                pfile = os.path.join(DIR, 'model/%d' % k, filename)
                gens[k].save_params(pfile)
