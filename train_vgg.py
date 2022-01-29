import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from loader import DataLoader
import logging
import os

def test_model(test_data, model):
    test_batch = test_data
    total, correct = 0, 0
    while test_batch.next():
        data, labels = test_batch.next()
        pred = model.predict(data)
        correct = np.sum(np.argmax(pred, axis=1) == labels)
        total += labels.shape[0]
    print(f'Accuracy at start: {correct/total}')

# Set configs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
img_height = 224
img_width = 224

# Build our augmentation
train_datagen = ImageDataGenerator(
                rescale=1./255,
                channel_shift_range = 0.2,
                horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load ImageNet dataset with the VGG augmentation
loader = DataLoader(batch_size=32, target_size=(img_height, img_width))
train_set, test_set = loader.load_ds_generator(train_datagen, test_datagen)

# Load VGG-16 with default as we are not transfer learning
model = tf.keras.applications.vgg16.VGG16(input_shape=(img_height, img_width, 3))
model.trainable = True

# Compile (not using original e-2 of paper)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss= tf.keras.losses.SparseCategoricalCrossentropy( ),
              metrics=['accuracy'])

# Test current accuracy on test-set
test_model(test_set, model)

# Train for 5 epochs
model.fit(train_set, epochs=5)

# Test new accuracy on test-set
test_model(test_set, model)