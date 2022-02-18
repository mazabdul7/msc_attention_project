import tensorflow as tf

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, init_vals=None, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.init_vals = init_vals

    def build(self, input_shape):
        self.weights = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )

    def call(self, inputs):
        return tf.multiply(inputs, self.w)

    def get_config(self):
        conf = super().get_config()
        conf.update({'input_filter_size': self.w.shape[-1]})

        return conf