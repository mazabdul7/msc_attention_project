import sys
gpu_dev = sys.argv[3]

import os
from random import sample
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Must be set before importing TF to supress messages
os.environ["CUDA_VISIBLE_DEVICES"]= gpu_dev

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
import numpy as np
from utils.loader import DataLoader
from utils.tools import test_model, insert_attention_layer_in_keras
from utils.data_sampler import CustomDataGenerator, CustomIterator
from utils.configs import config
from typing import List
import pymf
import pickle

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
    history = model.fit(train_set, validation_data=val_set, validation_freq=2, epochs=epochs, steps_per_epoch=train_set.n//batch_size, validation_steps=val_set.n//batch_size if val_set is not None else None, verbose=1, callbacks=callbacks)

    return history

if __name__ == '__main__':
    # Script takes args: fol_name, dimensionality, gpu
    fol_name = sys.argv[1]
    dims = int(sys.argv[2])


    # Set configs
    img_height = 224
    img_width = 224
    batch_size = 128
    epochs = 35
    lr = 5e-2
    base_path = os.path.join('models', 'task_models_fixed_test', fol_name, str(dims))
    os.makedirs(base_path)
    model_path = os.path.join(base_path, 'model_weights')
    log_path = os.path.join(base_path, 'log.csv')

    # Set augmentation and pre-processing
    train_datagen = CustomDataGenerator(
                    horizontal_flip=True,
                    validation_split=0.1,
                    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)
    test_datagen = CustomDataGenerator(
                    preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)

    # Load ImageNet dataset with the VGG augmentation
    loader = DataLoader(batch_size, (img_height, img_width))
    train_set = loader.load_train_set(aug_train=train_datagen, class_mode='categorical', shuffle=True)
    val_set = loader.load_val_set(aug_val=train_datagen, class_mode='categorical', shuffle=True)
    test_set = loader.load_test_set(aug_test=test_datagen, set_batch_size=False)

    # Load pre-trained VGG-16 model
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model('models/vgg_trained')
    model.trainable = False

    # Get projection matrix via SNMF
    with open('proj_mats_norm.pkl', 'rb') as f:
        p_dict  = pickle.load(f)
    p_mat = p_dict[dims]

    # Insert attention layer
    model = insert_attention_layer_in_keras(p_mat, model, ['block5_conv1'])


    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])


    # Enable subsampling equal to standard size
    train_set.set_target_sampling([fol_name], [0.5], force_class_sampling=True)


    # Train and use CSV logger to store logs
    if not os.path.exists(log_path):
        with open(log_path, "w") as my_empty_csv: pass

    csv_logger = CSVLogger(log_path, separator=',', append=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    def scheduler(epoch, lr):
        if epoch < 25:
            return lr
        else:
            return lr * tf.math.exp(-0.01)
    lr_sched = tf.keras.callbacks.LearningRateScheduler(scheduler)

    train_history = train_model(model=model, train_set=train_set, val_set=None, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger, early_stop, lr_sched])

    print('\nTraining Finished!')
    print(f'Saving Model Weights to {model_path}')
    model.save_weights(model_path)
    print('Weights Saved!')


    #test_model(model, test_set)
    # a = model.get_layer('attention_block5_conv1').seeds.numpy()
    # print(f'Sparsity in seed weights: {round(np.sum(a==0)/(a.shape[0]*a.shape[1]), 2)}')
