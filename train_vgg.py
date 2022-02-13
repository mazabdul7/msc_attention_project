import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Must be set before importing TF to supress messages
import tensorflow as tf
import numpy as np
from utils.loader import DataLoader
from utils.tools import test_model
from utils.data_sampler import CustomDataGenerator

# Set configs
img_height = 224
img_width = 224
batch_size = 16
epochs = 3

# Build our augmentation
train_datagen = CustomDataGenerator(
                rescale=1./255,
                channel_shift_range = 0.2,
                horizontal_flip=True,
                validation_split=0.2)
test_datagen = CustomDataGenerator(rescale=1./255)

# Load ImageNet dataset with the VGG augmentation
loader = DataLoader(batch_size, (img_height, img_width))
train_set, val_set, test_set = loader.load_ds_generator(train_datagen, test_datagen)
train_set.set_subsample(['n03873416'], [3/1000])
train_set._set_index_array()
#print(f'\nLabels are: {next(train_set)[1]}\n\nEntire index_arry for ds = {len(train_set.index_array)}') # Print labels

# Load VGG-16 with default as we are not transfer learning
model = tf.keras.applications.vgg16.VGG16(input_shape=(img_height, img_width, 3))
model.trainable = True

# Compile (not using original e-2 of paper)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss= tf.keras.losses.SparseCategoricalCrossentropy( ),
              metrics=['accuracy'])

# # Test current accuracy on test-set
test_model(model, test_set)

# Train for 5 epochs
model.fit(train_set, validation_data=val_set, epochs=epochs, steps_per_epoch=train_set.n//batch_size, validation_steps=val_set.n//batch_size)

# Test new accuracy on test-set
test_model(model, test_set)