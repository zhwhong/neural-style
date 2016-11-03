import glob
import importlib
import logging
import os
import random

import tensorflow as tf

from image_utils import *

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)

_ROOT = os.path.dirname(__file__)
DEFAULT_STYLE_IMAGE = os.path.join(_ROOT, 'data/style.jpg')
DEFAULT_CONTENT_DIR = os.path.join(_ROOT, 'data/contents')
DEFAULT_SUMMARY_DIR = os.path.join(_ROOT, 'logs')
DEFAULT_CHECKPOINT_DIR = os.path.join(_ROOT, 'ckpt-dir')
DEFAULT_TEXTURE_DIR = os.path.join(_ROOT, 'texture')

tf.app.flags.DEFINE_integer('image_size', 256, 'image size')
tf.app.flags.DEFINE_string('style', DEFAULT_STYLE_IMAGE, 'style image')
tf.app.flags.DEFINE_string('contents', DEFAULT_CONTENT_DIR,
                           'content image directory')
tf.app.flags.DEFINE_string('generator', 'g1', 'generator module')
tf.app.flags.DEFINE_integer('epoches', 100, 'training epoches')
tf.app.flags.DEFINE_integer('batch_size', 1, 'mini batch size')

tf.app.flags.DEFINE_string('device', '/cpu', 'training device')

tf.app.flags.DEFINE_bool('texture', False, 'texture or stylize')
tf.app.flags.DEFINE_float('learning_rate', 1., 'learning rate')
tf.app.flags.DEFINE_bool('train_weights', False, 'is weights trainable')
tf.app.flags.DEFINE_float('style_weight', 10., 'style weight')
tf.app.flags.DEFINE_float('content_weight', 1., 'content weight')
tf.app.flags.DEFINE_float('tv_weight', 0.01, 'tv weight')

tf.app.flags.DEFINE_string('content_layers',
                           '', 'content layer')
tf.app.flags.DEFINE_string('style_layers',
                           'relu1_2, relu2_2, relu3_2, relu4_2',
                           'style layers')

tf.app.flags.DEFINE_string('summary_dir', DEFAULT_SUMMARY_DIR,
                           'summary directory')
tf.app.flags.DEFINE_string('checkpoint_dir', DEFAULT_CHECKPOINT_DIR,
                           'checkpoint directory')
tf.app.flags.DEFINE_integer('snapshot_step', None, 'snapshot step')

tf.app.flags.DEFINE_string('texture_dir', DEFAULT_TEXTURE_DIR, 'texture_dir')

FLAGS = tf.app.flags.FLAGS

_STYLE_WEIGHT = FLAGS.style_weight
_CONTENT_WEIGHT = FLAGS.content_weight
_TV_WEIGHT = FLAGS.tv_weight
_STYLE_LAYERS = [s.strip() for s in str(FLAGS.style_layers).split(',')
                 if len(s.strip()) > 0]
if FLAGS.texture:
    _CONTENT_LAYERS = []
else:
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
        with tf.name_scope('style_gram'):
            image = tf.placeholder('float', shape=style_shape)
            net = vgg.net(image)
            for layer in _STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_image})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_gram[layer] = gram
    return style_gram


def _content_losses(x_features, content_features):
    with tf.name_scope('content_losses'):
        losses = {}
        for layer in _CONTENT_LAYERS:
            x_feature = x_features[layer]
            content_feature = content_features[layer]
            _, height, width, channel = map(lambda i: i.value,
                                            content_feature.get_shape())
            content_size = height * width * channel
            losses[layer] = tf.nn.l2_loss(
                x_feature - content_feature) / content_size
    return losses


def _style_losses(x_features, style_grams):
    with tf.name_scope('style_losses'):
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


def _tv_loss(image, batch_size):
    with tf.name_scope('tv_loss'):
        _, height, width, channel = map(lambda i: i.value,
                                        image.get_shape())
        tv_y_size = batch_size * (height - 1) * width * channel
        tv_x_size = batch_size * height * (width - 1) * channel
        y1 = tf.slice(image, [0, 1, 0, 0],
                      [batch_size, height - 1, width, channel],
                      name='y1')
        y0 = tf.slice(image, [0, 0, 0, 0],
                      [batch_size, height - 1, width, channel],
                      name='y0')
        y_loss = tf.nn.l2_loss(y1 - y0) / tv_y_size
        x1 = tf.slice(image, [0, 0, 1, 0],
                      [batch_size, height, width - 1, channel],
                      name='x1')
        x0 = tf.slice(image, [0, 0, 0, 0],
                      [batch_size, height, width - 1, channel],
                      name='x0')
        x_loss = tf.nn.l2_loss(x1 - x0) / tv_x_size
        loss = y_loss + x_loss
    return loss


def stylize(style_file, contents_dir, dshape, train_weights=False,
            generator_name='g1', learning_rate=0.001, n_epoches=10,
            batch_size=10,
            summary_dir='/tmp/tf_logs/', ckpt_dir=None, snapshot_step=None,
            texture_dir=None, device='/cpu'):
    # ===== generator
    if generator_name is None:
        generator_name = 'generator'
    logging.info("Generator: gens.%s" % generator_name)
    generator_class = importlib.import_module('gens.' + generator_name)

    # ===== define graph
    with tf.device(device):
        style_weight = tf.Variable(initial_value=_STYLE_WEIGHT,
                                   dtype=tf.float32, trainable=train_weights,
                                   name='style_weight')
        content_weight = tf.Variable(initial_value=_CONTENT_WEIGHT,
                                     dtype=tf.float32, trainable=train_weights,
                                     name='content_weight')
        tv_weight = tf.Variable(initial_value=_TV_WEIGHT, dtype=tf.float32,
                                trainable=train_weights,
                                name='tv_weight')

        X = tf.placeholder('float', shape=(None, dshape[0], dshape[1], 6))
        # styled images
        generator = generator_class.Generator(X, dshape)
        x_image = generator.net()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver(generator.variables + [global_step])
    saver.to_proto()

    assert len(dshape) == 3, 'dshape is not 3 dim'

    style_name = os.path.splitext(os.path.basename(style_file))[0]
    logging.info('Style: %s' % style_name)
    style_image = preprocess_image(style_file, dshape)

    # ===== checkpoint & summary
    if ckpt_dir is not None:
        ckpt_dir = os.path.join(ckpt_dir, style_name)
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
    logging.info('Checkpoint dir: %s' % ckpt_dir)
    if summary_dir is not None:
        summary_dir = os.path.join(summary_dir, style_name)
    logging.info('Summary dir: %s' % summary_dir)

    # ===== calculate style gram matrix
    style_grams = get_style_grams(style_image)

    # ===== calculate losses
    with tf.device(device):
        x_features = vgg.net(x_image)
        content_image, _ = tf.split(split_dim=3, num_split=2, value=X)
        content_features = vgg.net(content_image)
        with tf.name_scope('total_loss'):
            # loss
            content_losses = _content_losses(x_features, content_features)
            content_loss = 0
            for layer in _CONTENT_LAYERS:
                content_loss += content_losses[layer]
            style_losses = _style_losses(x_features, style_grams)
            style_loss = 0
            for layer in _STYLE_LAYERS:
                style_loss += style_losses[layer]
            tv_loss = _tv_loss(x_image, batch_size)
            total_loss = tf.mul(content_weight, content_loss) \
                         + tf.mul(style_weight, style_loss) \
                         + tf.mul(tv_weight, tv_loss)

        # optimizer set
        logging.info("Learning rate: %g" % learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # ===== summary
    # weights
    tf.histogram_summary('style_weight', style_weight, name='style_weight')
    tf.histogram_summary('content_weight', content_weight,
                         name='content_weight')
    tf.histogram_summary('tv_weight', tv_weight, name='tv_weight')

    # layer loss
    for layer in _STYLE_LAYERS:
        tf.scalar_summary('style-loss[%s]' % layer, style_losses[layer])
    for layer in _CONTENT_LAYERS:
        tf.scalar_summary('content-loss[%s]' % layer, content_losses[layer])

    # overall loss
    tf.scalar_summary('content-loss', content_loss)
    tf.scalar_summary('style-loss', style_loss)
    tf.scalar_summary('tv-loss', tv_loss)
    tf.scalar_summary('total-loss', total_loss)

    # ===== load images
    filelist = [f for f in glob.glob(contents_dir + "/*")
                if os.path.splitext(f.lower())[1][1:] in ('jpg', 'jpeg')]
    n_images = len(filelist)
    logging.info('Content images size: %d' % n_images)
    images = [preprocess_image(f, dshape) for f in filelist]

    # ===== train
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(summary_dir, sess.graph) \
            if summary_dir is not None else None

        sess.run(tf.initialize_all_variables())

        if ckpt_dir is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt is not None and ckpt.model_checkpoint_path is not None:
                logging.info(
                    'Restore session from: %s' % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

        start_step = global_step.eval()
        logging.info("start epoch: %d" % start_step)

        # iterations
        best_loss = float('inf')
        best_img = None
        best_step = 0
        for epoch in range(start_step, start_step + n_epoches):
            global_step.assign(epoch).eval()
            random.shuffle(images)

            batch_steps = n_images / batch_size
            for i in range(batch_steps):
                feed = np.append(images[batch_size * i:batch_size * (i + 1)],
                                 np.random.normal(size=(batch_size,) + dshape),
                                 axis=3)
                _, x_img, loss, summary = sess.run(
                    [train_step, x_image, total_loss, merged],
                    feed_dict={X: feed})
                logging.info("epoch=%d, batch=%d, loss:%g" % (epoch, i, loss))
                if summary_dir is not None:
                    writer.add_summary(summary, epoch * n_images + i)
                if loss < best_loss:
                    best_loss = loss
                    best_img = x_img
                    best_step = epoch

            if texture_dir is not None:
                if snapshot_step is not None and epoch % snapshot_step == 0 \
                        or epoch == start_step + n_epoches - 1:
                    noise = _NOISE_IMAGE.reshape(
                        (1,) + _NOISE_IMAGE.shape) - vgg.MEAN_PIXEL
                    feed = np.append(noise, np.random.normal(size=noise.shape),
                                     axis=3) * 0.1
                    img = sess.run(x_image, feed_dict={X: feed})
                    img = img.reshape(dshape) + vgg.MEAN_PIXEL
                    texture_path = os.path.join(texture_dir, '%s-%d.jpg' % (
                        style_name, epoch))
                    imsave(texture_path, img)

            if ckpt_dir is not None:
                if snapshot_step is not None and epoch % snapshot_step == 0 \
                        or epoch == start_step + n_epoches - 1:
                    model_path = os.path.join(ckpt_dir, "%s.model" % style_name)
                    logging.info(
                        'Saving checkpoint to: %s-%d' % (model_path, epoch))
                    saver.save(sess, model_path, global_step=global_step)
        logging.info("best loss @ step %d: %g" % (best_step, best_loss))
        for i in range(batch_size):
            img = np.reshape(best_img[i, :, :, :], dshape)
            path = os.path.join(texture_dir, 'best-%s-%d-%d.jpg' % (
                style_name, best_step, i))
            imsave(path, img)


def main():
    dshape = (FLAGS.image_size, FLAGS.image_size, 3)
    global _NOISE_IMAGE
    _NOISE_IMAGE = np.random.normal(size=dshape)

    stylize(style_file=FLAGS.style,
            contents_dir=FLAGS.contents,
            dshape=dshape,
            generator_name=FLAGS.generator,
            learning_rate=FLAGS.learning_rate,
            n_epoches=FLAGS.epoches,
            batch_size=FLAGS.batch_size,
            summary_dir=FLAGS.summary_dir,
            ckpt_dir=FLAGS.checkpoint_dir,
            snapshot_step=FLAGS.snapshot_step,
            device=FLAGS.device,
            texture_dir=FLAGS.texture_dir
            )


if __name__ == '__main__':
    main()