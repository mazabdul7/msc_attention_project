import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Must be set before importing TF to supress messages
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
import tensorflow as tf
import numpy as np
from utils.loader import DataLoader
from utils.tools import test_model
from utils.data_sampler import CustomDataGenerator

# Set configs
img_height = 224
img_width = 224
batch_size = 64
epochs = 3

# Build our augmentation
train_datagen = CustomDataGenerator(
                channel_shift_range = 0.2,
                horizontal_flip=True,
                validation_split=0.2,
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)
test_datagen = CustomDataGenerator(
                preprocessing_function=tf.keras.applications.vgg16.preprocess_input, dtype=tf.float32)

# Load ImageNet dataset with the VGG augmentation
loader = DataLoader(batch_size, (img_height, img_width))
train_set, val_set, test_set = loader.load_ds_generator(train_datagen, test_datagen)
train_set.set_subsample(['n03873416'], [5/1000]) # No subsampling
train_set._set_index_array()

# Load VGG-16 with default as we are not transfer learning
model = tf.keras.applications.vgg16.VGG16(input_shape=(img_height, img_width, 3))
model.trainable = True

# Compile (not using original e-2 of paper)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss= tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Test current accuracy on test-set (should be 76.3)
test_model(model, test_set)

# Train for 5 epochs
model.fit(train_set, validation_data=val_set, epochs=epochs, steps_per_epoch=train_set.n//batch_size, validation_steps=val_set.n//batch_size, verbose=1)

# Test new accuracy on test-set
test_model(model, test_set)