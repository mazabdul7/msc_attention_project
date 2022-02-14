#
# Loader that returns a generator 
#
import os
from utils.data_sampler import CustomDataGenerator, CustomIterator
from typing import Tuple
import random

class DataLoader:
    def __init__(self, batch_size: int, target_size: Tuple[int]) -> None:
        # Paths relative to working directory
        self.base_path = r'/fast-data22/datasets/ILSVRC/2012/clsloc'
        self.train_path = r'train'
        self.test_path = r'val_white'

        # Configs
        self.batch_size = batch_size
        self.target_size = target_size
        self.seed = random.randint(0, 1000)

    def load_ds_generator(self, aug_train: CustomDataGenerator, aug_test: CustomDataGenerator) -> CustomIterator:
        ''' 
            Loads and returns batched tf.Dataset generator object from passed path. 
            If val: loads validation set. 
        '''
        print('Loading train set...')
        train_generator = aug_train.flow_from_directory(
                os.path.join(self.base_path, self.train_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='sparse',
                seed=self.seed,
                subset='training')
        
        print('Loading validation set...')
        val_generator = aug_train.flow_from_directory(
                os.path.join(self.base_path, self.train_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode='sparse',
                seed=self.seed,
                subset='validation')
        
        print('Loading test set...')
        test_generator = aug_test.flow_from_directory(
                os.path.join(self.base_path, self.test_path),
                target_size=self.target_size,
                batch_size=1,
                class_mode=None,
                shuffle=False)

        return train_generator, val_generator, test_generator