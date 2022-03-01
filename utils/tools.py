from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from utils.data_sampler import CustomIterator
from models.layers import AttentionLayer

def test_model(model: Model, test_set: CustomIterator) -> None:
    """ Test the passed model for its Top-1 accuracy on the passed test set.

    Args:
        model (Model): TensorFlow model.
        test_set (CustomIterator): Test set to test on.
    """
    print('\nPredicting on test-set...')
    test_set.reset()
    pred = model.predict(test_set, steps=test_set.n//test_set.batch_size, verbose=1)
    pred = np.argmax(pred, axis=-1)
    cls = np.array(test_set.classes)
    
    print('Computing accuracy...')
    accuracy = np.sum(pred == cls)/len(pred)
    print('\n-----------------------------------------')
    print(f'Model Accuracy on test-set: {accuracy}')
    print('-----------------------------------------\n')

def insert_attention_layer_in_keras(model: Model, layer_names: List[str]) -> Model:
    """ Insert an attention layer preceeding the passed layer name within the model

    Args:
        model (Model): TensorFlow model.
        layer_name (List[str]): List of layer names to insert the attention layers at.

    Returns:
        Model: TensorFlow model.
    """
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if layers[i]._name in layer_names:
            x = AttentionLayer(name='attention_' + layers[i]._name)(x)
        x = layers[i](x)

    new_model = tf.keras.models.Model(inputs=layers[0].input, outputs=x)
    return new_model