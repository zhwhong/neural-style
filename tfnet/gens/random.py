import tensorflow as tf


class Generator(object):
    def __init__(self, X, dshape):
        self._height = dshape[0]
        self._width = dshape[1]
        self._variables = self._init_variables()

    def _init_variables(self):
        return {
            'random': tf.Variable(
                tf.random_normal([1, self._height, self._width, 3]) * 0.256,
                name='random')
        }

    @property
    def variables(self):
        return self._variables.values()

    def net(self):
        return self._variables['random']