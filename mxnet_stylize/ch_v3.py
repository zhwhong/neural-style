# coding: utf-8

import mxnet as mx


def Conv(data, num_filter, kernel=(5, 5), pad=(2, 2), stride=(2, 2)):
    sym = mx.sym.Convolution(data, num_filter=num_filter, kernel=kernel,
                             stride=stride, pad=pad, no_bias=False)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    return sym


def Deconv(data, num_filter, im_hw, kernel=(7, 7), pad=(2, 2), stride=(2, 2),
           crop=True, out=False):
    sym = mx.sym.Deconvolution(data, num_filter=num_filter, kernel=kernel,
                               stride=stride, pad=pad, no_bias=True)
    if crop:
        sym = mx.sym.Crop(sym, offset=(1, 1), h_w=im_hw, num_args=1)
    sym = mx.sym.BatchNorm(sym, fix_gamma=False)
    if out == False:
        sym = mx.sym.LeakyReLU(sym, act_type="leaky")
    else:
        sym = mx.sym.Activation(sym, act_type="tanh")
    return sym


# In[70]:

def get_generator(prefix, im_hw):
    data = mx.sym.Variable("%s_data" % prefix)
    conv6_1 = mx.sym.Convolution(data, num_filter=32, kernel=(5, 5), pad=(2, 2), no_bias=True)
    conv6_1 = mx.sym.BatchNorm(conv6_1, fix_gamma=False)
    conv6_1 = mx.sym.LeakyReLU(conv6_1, act_type="leaky")

    out = mx.sym.Convolution(conv6_1, num_filter=3, kernel=(3, 3), pad=(1, 1),
                             no_bias=True)
    out = mx.sym.BatchNorm(out, fix_gamma=False)
    out = mx.sym.Activation(data=out, act_type="tanh")

    raw_out = (out * 128) + 128
    norm = mx.sym.SliceChannel(raw_out, num_outputs=3)
    r_ch = norm[0] - 123.68
    g_ch = norm[1] - 116.779
    b_ch = norm[2] - 103.939
    norm_out = 0.4 * mx.sym.Concat(*[r_ch, g_ch, b_ch]) + 0.6 * data
    return norm_out


def get_module(prefix, dshape, ctx, is_train=True):
    sym = get_generator(prefix, dshape[-2:])
    # print 'sym'
    # print sym.tojson()
    mod = mx.mod.Module(symbol=sym,
                        data_names=("%s_data" % prefix,),
                        label_names=None,
                        context=ctx)
    if is_train:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=True,
                 inputs_need_grad=True)
    else:
        mod.bind(data_shapes=[("%s_data" % prefix, dshape)], for_training=False,
                 inputs_need_grad=False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    return mod
