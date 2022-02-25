import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals

    def build(self, input_shape):
        """ Initialises attention weights of shape equivalent to the input channels shape for per channel scaling.
        """
        self.kernel = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
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