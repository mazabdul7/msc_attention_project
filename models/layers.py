import tensorflow as tf
import numpy as np

class NonNegConstraint(tf.keras.constraints.Constraint):
    """Constrains seed tensors to ensure projected weights do not go below 0."""

    def __init__(self, proj_mat):
        self.proj_mat = proj_mat
        self.min_value = -1
        self.epsilon = 1e-7

    def __call__(self, w):
        projected_attention = tf.matmul(w, self.proj_mat)
        desired = tf.clip_by_value(projected_attention, self.min_value, np.inf)

        return w * tf.reduce_min((desired / (projected_attention - self.epsilon)))

class AttentionLayer(tf.keras.layers.Layer):
    """ Regular attention layer (Luo et al. 2021)
    """
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals

    def build(self, input_shape):
        """ Initialises attention weights of shape equivalent to the input channels shape for per channel scaling.
        """
        self.kernel = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
            name='weights'
        )

    def call(self, inputs):
        """ Forward pass performs a depthwise convolution of the attention weights while mixing no channel information.

            Note: The kernel must be expanded to follow [filter_height, filter_width, in_channels, channel_multiplier] where
            the in_channels can be substituted for a same size attention weight scaling per channel axis, hence performing per channel
            scaling by the attention weights.
        """
        return tf.nn.depthwise_conv2d(inputs, self.kernel[tf.newaxis, tf.newaxis, :, tf.newaxis], [1, 1, 1, 1], 'VALID')

    def get_config(self):
        conf = super().get_config()
        conf.update({'input_filter_size': self.kernel.shape[-1]})

        return conf

class ProjectionAttentionLayer(tf.keras.layers.Layer):
    """ Latent attention seed layer that projects via an internal step-up matrix.
    """
    def __init__(self, p_mat, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals
        self.projection_mat = p_mat

    def build(self, input_shape):
        """ Initialises attention seeds and bias for affine projection into latent space.
        """
        self.seeds = self.add_weight(
            shape=(1, self.projection_mat.shape[0],),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            constraint=NonNegConstraint(self.projection_mat),
            name='seeds'
        )
        self.bias = self.add_weight(
            shape=(1, input_shape[-1],),
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name='bias'
        )

    def call(self, inputs):
        """ Forward pass performs a depthwise convolution of the attention weights projected via seeding method while mixing no channel information.

            Note: The kernel must be expanded to follow [filter_height, filter_width, in_channels, channel_multiplier] where
            the in_channels can be substituted for a same size attention weight scaling per channel axis, hence performing per channel
            scaling by the attention weights.
        """
        projected_attention = tf.squeeze(tf.matmul(self.seeds, self.projection_mat) + self.bias)
        return tf.nn.depthwise_conv2d(inputs, projected_attention[tf.newaxis, tf.newaxis, :, tf.newaxis], [1, 1, 1, 1], 'VALID')

    def get_config(self):
        conf = super().get_config()
        conf.update({'input_filter_size': self.kernel.shape[-1]})

        return conf