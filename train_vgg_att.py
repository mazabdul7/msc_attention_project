import os
from random import sample
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Must be set before importing TF to supress messages
os.environ["CUDA_VISIBLE_DEVICES"]= '2'

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
from utils.loader import DataLoader
from utils.tools import test_model
from utils.data_sampler import CustomDataGenerator, CustomIterator
from utils.configs import config
from models.layers import AttentionLayer
from typing import List

def load_VGG_model(img_height: int, img_width: int, lr: int, loss: tf.keras.losses.Loss, metrics: List[str], trainable: True) -> tf.keras.Model:
    """ Loads VGG-16 model.

    Args:
        img_height (int): Image height.
        img_width (int): Image width.
        lr (int): Learning rate.
        loss (tf.keras.losses.Loss): Model loss.
        metrics (List[str]): Training metrics.
        trainable (True): Set if model weights should be kept frozen or not.

    Returns:
        tf.keras.Model: TensorFlow VGG-16 model.
    """
    model = tf.keras.applications.vgg16.VGG16(input_shape=(img_height, img_width, 3))
    model.trainable = trainable
    model.compile(optimizer=tf.keras.optimizers.SGD(lr),
                loss=loss,
                metrics=metrics)

    return model

def insert_attention_layer_in_keras(model, layer_name):
    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if layers[i]._name == layer_name:
            x = AttentionLayer(name='attention_layer')(x)
        x = layers[i](x)

    new_model = tf.keras.models.Model(inputs=layers[0].input, outputs=x)
    return new_model

def train_model(model: tf.keras.Model, train_set: CustomIterator, val_set: CustomIterator, epochs: int, batch_size: int, callbacks=None):
    """ Train the model. 

    Args:
        train_set (CustomIterator): Training data.
        val_set (CustomIterator): Validation data.
        epochs (int): Number of epochs to train for.
        callbacks (_type_, optional): Callbacks for loggers. Defaults to None.

    Returns:
        history: Model training history information.
    """
    history = model.fit(train_set, validation_data=val_set, epochs=epochs, steps_per_epoch=train_set.n//batch_size, validation_steps=val_set.n//batch_size, verbose=1, callbacks=callbacks)

    return history

def main(img_height: int, img_width: int, batch_size: int, lr: int, epochs: int, set_subsample: bool=False, sample_size: int= None, target_classes: List[str]=None, target_weights: List[int]=None) -> None:
    """ Main training loop. """
    print(tf.__version__)
    # Set configs
    img_height = img_height
    img_width = img_width
    batch_size = batch_size
    epochs = epochs
    lr = lr
    log_path = os.path.join(config['logs_path'], 'vgg_training_att.csv')

    # Set augmentation and pre-processing
    train_datagen = CustomDataGenerator(
                    channel_shift_range = 0.2,
                    horizontal_flip=True,
                    validation_split=0.2,
                    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)
    test_datagen = CustomDataGenerator(
                    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)

    # Load ImageNet dataset with the VGG augmentation
    loader = DataLoader(batch_size, (img_height, img_width))
    train_set = loader.load_train_set(aug_train=train_datagen, class_mode='sparse', shuffle=True)
    val_set = loader.load_val_set(aug_val=train_datagen, class_mode='sparse', shuffle=True)
    test_set = loader.load_test_set(aug_test=test_datagen, set_batch_size=False)

    if set_subsample:
        # Enable sub-sampling to get smaller ImageNet set for faster training
        if target_classes and target_weights:
            train_set.set_target_sampling(target_classes=target_classes, target_weights=target_weights)
        elif sample_size:
            train_set.set_subsampling(sample_size)

    # Load VGG-16 with default as we are not transfer learning
    model = load_VGG_model(img_height=img_height, img_width=img_width, lr=lr, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'], trainable=True)
    
    # Insert attention layer before last block
    model = insert_attention_layer_in_keras(model, 'block5_conv1')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    model.summary()

    # Test current accuracy on test-set
    #test_model(model, test_set)

    # Train and use CSV logger to store logs
    if not os.path.exists(log_path):
        with open(log_path, "w") as my_empty_csv: pass

    csv_logger = CSVLogger(log_path, separator=',', append=False)
    train_history = train_model(model=model, train_set=train_set, val_set=val_set, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger])
    print(model.get_layer('attention_layer').get_weights())

    # Test new accuracy on test-set
    test_model(model, test_set)

if __name__ == '__main__':
    main(img_height=224, img_width=224, batch_size=32, lr=1e-4, epochs=1, set_subsample=True, sample_size=500000)