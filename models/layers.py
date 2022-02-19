import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1],),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.kernel[tf.newaxis, tf.newaxis, :, tf.newaxis], [1, 1, 1, 1], 'VALID')

    def get_config(self):
        conf = super().get_config()
        conf.update({'input_filter_size': self.w.shape[-1]})

        return conf