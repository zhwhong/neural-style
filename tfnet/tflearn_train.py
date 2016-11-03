import cPickle
import importlib
import os
import shutil
import time

import numpy as np
import tensorflow as tf
import tflearn
from skimage import io

import vgg
from image_utils import preprocess_image

_RUN_ID = 'chwang'

shutil.rmtree('/tmp/tflearn_logs/%s' % _RUN_ID, ignore_errors=True)

_ROOT = os.path.dirname(__file__)
DEFAULT_STYLE_IMAGE = os.path.join(_ROOT, 'data/style.jpg')
DEFAULT_CONTENT_DIR = os.path.join(_ROOT, 'data/contents')

tf.app.flags.DEFINE_integer('image_size', 256, 'image size')
tf.app.flags.DEFINE_string('style', DEFAULT_STYLE_IMAGE, 'style image')
tf.app.flags.DEFINE_string('contents', DEFAULT_CONTENT_DIR,
                           'content image directory')
tf.app.flags.DEFINE_string('generator', 'g1', 'generator module')
tf.app.flags.DEFINE_integer('epoches', 10, 'training epoches')
tf.app.flags.DEFINE_string('device', '/cpu', 'training device')
tf.app.flags.DEFINE_bool('train', True, 'is train')
tf.app.flags.DEFINE_string('prefix', '', 'output prefix')
tf.app.flags.DEFINE_string('content_layer', 'relu3_1', 'content layer')
tf.app.flags.DEFINE_string('style_layers',
                           'relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu4_1,relu5_1',
                           'style layers')
tf.app.flags.DEFINE_string('style_layer_weights',
                           '1,1,1,1,1,1,1,1,1,1',
                           'style layer weights')
FLAGS = tf.app.flags.FLAGS

_STYLE_LAYERS = [s.strip() for s in str(FLAGS.style_layers).split(',')]
_STYLE_LAYER_WEIGHTS = [float(s.strip()) for s in
                        str(FLAGS.style_layer_weights).split(',')]
_CONTENT_LAYER = FLAGS.content_layer

_NOISE_WEIGHT = 1e-1
_STYLE_WEIGHT = 2e1
_CONTENT_WEIGHT = 1e1
_TV_WEIGHT = 0


# style_weight = tf.Variable(initial_value=1e2, dtype=tf.float32)
# content_weight = tf.Variable(initial_value=1e1, dtype=tf.float32)


def _style_loss(generated_features, style_features):
    style_loss = 0.
    for i in range(len(_STYLE_LAYERS)):
        layer = _STYLE_LAYERS[i]
        weight = _STYLE_LAYER_WEIGHTS[i]
        style_feature = style_features[layer]
        generated_feature = generated_features[layer]
        _, height, width, channel = map(lambda i: i.value,
                                        style_feature.get_shape())
        size = height * width * channel
        s_features = tf.reshape(style_feature, (-1, channel))
        s_gram = tf.matmul(tf.transpose(s_features), s_features) / size
        g_features = tf.reshape(generated_feature, (-1, channel))
        g_gram = tf.matmul(tf.transpose(g_features), g_features) / size
        style_loss += weight * tf.nn.l2_loss(g_gram - s_gram) / size
    return style_loss


def _content_loss(generated_features, content_features):
    content_feature = content_features[_CONTENT_LAYER]
    generated_feature = generated_features[_CONTENT_LAYER]
    _, height, width, channel = map(lambda i: i.value,
                                    content_feature.get_shape())
    size = height * width * channel
    loss = tf.nn.l2_loss(generated_feature - content_feature) / size
    return loss


def _tv_loss(image):
    batch, h, w, c = map(lambda i: i.value, image.get_shape())
    loss = 0
    imgs = tf.split(split_dim=0, num_split=batch, value=image)
    for img in imgs:
        y = tf.image.crop_to_bounding_box(img, 1, 0, h - 1, w)
        x = tf.image.crop_to_bounding_box(img, 0, 1, h, w - 1)
        tv_y_size = (h - 1) * w * c
        tv_x_size = h * (w - 1) * c
        y_diff = y - tf.image.crop_to_bounding_box(img, 0, 0, h - 1, w)
        x_diff = x - tf.image.crop_to_bounding_box(img, 0, 0, h, w - 1)

        loss += (tf.nn.l2_loss(y_diff) / tv_y_size) + (
            tf.nn.l2_loss(x_diff) / tv_x_size)
    return loss


def load_pkl(path):
    with open(path, 'r') as f:
        return cPickle.load(f)


def total_loss(y_pred, y_true):
    loss = 0
    generative_net = y_pred
    content, style = tf.split(split_dim=3, num_split=2, value=y_true)
    generated_features = vgg.net(generative_net)
    if _STYLE_WEIGHT > 0:
        style_features = vgg.net(style)
        style_loss = _style_loss(generated_features, style_features)
        loss += _STYLE_WEIGHT * style_loss
    if _CONTENT_WEIGHT > 0:
        content_features = vgg.net(content)
        content_loss = _content_loss(generated_features, content_features)
        loss += _CONTENT_WEIGHT * content_loss
    if _TV_WEIGHT > 0:
        loss += _TV_WEIGHT * _tv_loss(generative_net)
    return loss


def get_training_data(content_path, style_image, dshape, batch=0,
                      batch_size=100):
    X = []
    Y = []
    assert os.path.isdir(content_path)
    content_list = [f for f in os.listdir(content_path)
                    if os.path.splitext(f.lower())[1][1:] in ('jpg', 'jpeg')]
    start = batch * batch_size
    end = min((batch + 1) * batch_size, len(content_list))
    for i in range(start, end):
        filename = content_list[i]
        file_path = content_path + '/' + filename
        content_image = preprocess_image(file_path, dshape)
        noise_image = np.random.normal(size=[dshape[0], dshape[1], 3])
        noise_image *= _NOISE_WEIGHT
        X.append(np.concatenate((content_image, noise_image), axis=2))
        Y.append(np.concatenate((content_image, style_image), axis=2))
    return X, Y


def save_styled_image(image, content_name, style_name, prefix=''):
    if not os.path.exists('output'):
        os.makedirs('output')
    output_path = os.path.join('output', '%s_%s_%s.jpg' % (
        content_name, style_name, prefix))
    io.imsave(output_path, image)


def train(style_file, content_path, generator_name=None, n_iter=10, n_ckpt=3,
          dshape=None, device='/cpu', prefix=''):
    print('style: %s' % style_file)
    if generator_name is None:
        generator_name = 'generator'
    generator = importlib.import_module(generator_name)

    start_time = time.time()
    style_name = os.path.splitext(os.path.basename(style_file))[0]
    style_image = preprocess_image(style_file, dshape)

    # Define model
    with tf.device(device):
        y = tf.placeholder(tf.float32, shape=[None, dshape[0], dshape[1], 6])
        x = tflearn.input_data(shape=[None, dshape[0], dshape[1], 6])
        generative_net = generator.net(x)
        net = tflearn.regression(generative_net, placeholder=y,
                                 loss=total_loss, metric=None)
    model = tflearn.DNN(net, checkpoint_path=os.path.join(_ROOT, 'models',
                                                          '%s.model' % style_name))

    # Load if exists
    model_file = os.path.join('models', '%s.model' % style_name)
    if os.path.isfile(model_file):
        model.load(model_file)

    data_batch = 0
    while (True):
        X, Y = get_training_data(content_path, style_image, dshape,
                                 batch=data_batch)
        if len(X) == 0:
            break
        model.fit(X, Y, n_epoch=n_iter, snapshot_epoch=False,
                  batch_size=1, run_id=_RUN_ID)
        data_batch += 1
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(model_file)
    model.save(model_file + "_%s" % prefix)
    print('Train time: %d' % (time.time() - start_time))


def main():
    dshape = [FLAGS.image_size, FLAGS.image_size, 3]
    train(FLAGS.style, FLAGS.contents, dshape=dshape,
          generator_name=FLAGS.generator, n_iter=FLAGS.epoches,
          device=FLAGS.device, prefix=FLAGS.prefix)


if __name__ == '__main__':
    main()
