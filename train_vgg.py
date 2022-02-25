import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Must be set before importing TF to supress messages
os.environ["CUDA_VISIBLE_DEVICES"]= '3'

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
from utils.loader import DataLoader
from utils.tools import test_model
from utils.data_sampler import CustomDataGenerator, CustomIterator
from utils.configs import config
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, epsilon=0.1),
                loss=loss,
                metrics=metrics)

    return model

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
    # Set configs
    img_height = img_height
    img_width = img_width
    batch_size = batch_size
    epochs = epochs
    lr = lr
    log_path = os.path.join(config['logs_path'], 'vgg_training.csv')

    # Set augmentation and pre-processing
    train_datagen = CustomDataGenerator(
                    horizontal_flip=True,
                    validation_split=0.1,
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

    # Test current accuracy on test-set
    test_model(model, test_set)

    # Train and use CSV logger to store logs
    if not os.path.exists(log_path):
        with open(log_path, "w") as my_empty_csv: pass

    csv_logger = CSVLogger(log_path, separator=',', append=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    train_history = train_model(model=model, train_set=train_set, val_set=val_set, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger, early_stop])

    # Test new accuracy on test-set
    test_model(model, test_set)

if __name__ == '__main__':
    main(img_height=224, img_width=224, batch_size=64, lr=1e-7, epochs=20, set_subsample=True, sample_size=250000)