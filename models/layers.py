import tensorflow as tf
import numpy as np

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals

    def build(self, input_shape):
        self.w = tf.Variable(
            initial_value=np.ones(shape=(1, input_shape[-1])),
            trainable=self.trainable
        )

    def call(self, inputs):
        return tf.multiply(inputs, self.w)

    def get_config(self):
        return {'input_filter_size': self.w.shape[-1], 'weights': self.w}