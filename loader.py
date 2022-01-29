#
# Loader that returns a generator 
#
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Tuple, Iterator

class DataLoader:
    def __init__(self, batch_size: int, target_size: Tuple[int]) -> None:
        # ImageNet path within server
        self.base_path = r'/fast-data22/datasets/daniel/0_imagenet_split'
        self.train_path = r'train'
        self.test_path = r'val'

        # Configs
        self.batch_size = batch_size
        self.target_size = target_size

    def load_ds_generator(self, aug_train: ImageDataGenerator, aug_test: ImageDataGenerator) -> Iterator:
        ''' 
            Loads and returns batched tf.Dataset generator object from passed path. 
            If val: loads validation set. 
        '''
        print('Loading train set...')
        train_generator = aug_train.flow_from_directory(
                os.path.join(self.base_path, self.train_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='sparse')
        
        print('Loading test set...')
        validation_generator = aug_test.flow_from_directory(
                os.path.join(self.base_path, self.test_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='sparse')

        return train_generator, validation_generator