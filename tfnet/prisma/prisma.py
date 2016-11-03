import glob
import logging
import os

import tensorflow as tf

from image_utils import *

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

_ROOT = os.path.dirname(__file__)
_PRISMA = os.path.join(_ROOT, 'data', 'prisma')

DEFAULT_STYLE_IMAGE = os.path.join(_ROOT, 'data/style.jpg')
DEFAULT_CONTENT_DIR = os.path.join(_ROOT, 'data/contents')
DEFAULT_SUMMARY_DIR = os.path.join(_ROOT, 'logs')
DEFAULT_CHECKPOINT_DIR = os.path.join(_ROOT, 'ckpt-dir')
DEFAULT_TEXTURE_DIR = os.path.join(_ROOT, 'texture')

tf.app.flags.DEFINE_integer('image_size', 256, 'image size')
tf.app.flags.DEFINE_string('s', '', 'style')
tf.app.flags.DEFINE_string('c', '', 'content')
tf.app.flags.DEFINE_string('p', '', 'prisma')
tf.app.flags.DEFINE_string('device', '/cpu', 'training device')

tf.app.flags.DEFINE_string('content_layers',
                           'relu3_1, relu4_2', 'content layer')
tf.app.flags.DEFINE_string('style_layers',
                           'relu1_1, relu1_2, relu2_1, relu2_2, relu3_1, relu4_1, relu5_1',
                           'style layers')
FLAGS = tf.app.flags.FLAGS

_STYLE_LAYERS = [s.strip() for s in str(FLAGS.style_layers).split(',')
                 if len(s.strip()) > 0]
_CONTENT_LAYERS = [s.strip() for s in str(FLAGS.content_layers).split(',')
                   if len(s.strip()) > 0]
logging.info("Style layers: %s" % str(_STYLE_LAYERS))
logging.info("Content layers: %s" % str(_CONTENT_LAYERS))


def get_style_grams(style_image):
    style_gram = {}
    style_shape = (1,) + style_image.shape
    assert len(style_shape) == 4, 'Style image is not 3 dim'
    style_image = np.reshape(style_image, style_shape)
    # compute style features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu'), tf.Session(graph=g):
        image = tf.placeholder('float', shape=style_shape)
        # print "style-shape", image
        net = vgg.net(image)
        for layer in _STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_image})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_gram[layer] = gram
    return style_gram


def _content_losses(x_features, content_features):
    losses = {}
    for layer in _CONTENT_LAYERS:
        if len(layer) == 0:
            continue
        x_feature = x_features[layer]
        content_feature = content_features[layer]
        _, height, width, channel = map(lambda i: i.value,
                                        content_feature.get_shape())
        content_size = height * width * channel
        losses[layer] = tf.nn.l2_loss(
            x_feature - content_feature) / content_size
    return losses


def _style_losses(x_features, style_grams):
    losses = {}
    for layer in _STYLE_LAYERS:
        x_feature = x_features[layer]
        _, height, width, channel = map(lambda i: i.value,
                                        x_feature.get_shape())
        size = height * width * channel
        x_feats = tf.reshape(x_feature, (-1, channel))
        x_gram = tf.matmul(tf.transpose(x_feats), x_feats) / size
        style_gram = style_grams[layer]
        losses[layer] = tf.nn.l2_loss(x_gram - style_gram) / style_gram.size
    return losses


def _tv_loss(image, dshape, batch_size=1):
    _, height, width, channel = map(lambda i: i.value,
                                    image.get_shape())
    tv_y_size = batch_size * (height - 1) * width * channel
    tv_x_size = batch_size * height * (width - 1) * channel
    y_loss = tf.nn.l2_loss(
        image[:, 1:, :, :] - image[:, :dshape[0] - 1, :, :]) / tv_y_size
    x_loss = tf.nn.l2_loss(
        image[:, :, 1:, :] - image[:, :, :dshape[1] - 1, :]) / tv_x_size
    loss = y_loss + x_loss
    return loss


def stylize(style_file, content_file, prisma_file, dshape, device='/cpu'):
    assert len(dshape) == 3, 'dshape is not 3 dim'

    style_name = os.path.splitext(os.path.basename(style_file))[0]
    style_image = preprocess_image(style_file, dshape)
    content_image = preprocess_image(content_file, dshape)
    prisma_image = preprocess_image(prisma_file, dshape)

    # ===== calculate style gram matrix
    style_grams = get_style_grams(style_image)

    # ===== define graph
    with tf.device(device):
        content = tf.constant(content_image, dtype=tf.float32, shape=(1,) + content_image.shape)
        x_image = tf.constant(prisma_image, dtype=tf.float32, shape=(1,) + prisma_image.shape)
        content_features = vgg.net(content)
        x_features = vgg.net(x_image)

        # loss
        content_losses = _content_losses(x_features, content_features)
        content_loss = 0
        for layer in _CONTENT_LAYERS:
            content_loss += content_losses[layer]
        style_losses = _style_losses(x_features, style_grams)
        style_loss = 0
        for layer in _STYLE_LAYERS:
            style_loss += style_losses[layer]
        tv_loss = _tv_loss(x_image, dshape)

    # eval
    with tf.Session():
        for layer in _STYLE_LAYERS:
            logging.info("Style loss @ %s : %g" % (layer, style_losses[layer].eval()))
        for layer in _CONTENT_LAYERS:
            logging.info(
                "Content loss @ %s : %g" % (
                    layer, content_losses[layer].eval()))
        logging.info("Style loss: %g" % (style_loss.eval()))
        logging.info("Content loss: %g" % (content_loss.eval()))
        logging.info("TV loss: %g" % (tv_loss.eval()))


def main():
    dshape = (FLAGS.image_size, FLAGS.image_size, 3)
    device = FLAGS.device
    logging.info("Style: %s" % FLAGS.s)
    logging.info("Content: %s" % FLAGS.c)
    stylize(FLAGS.s, FLAGS.c, FLAGS.p, dshape, device)


if __name__ == '__main__':
    main()