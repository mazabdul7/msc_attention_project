import os
from utils.data_sampler import CustomDataGenerator, CustomIterator
from utils.configs import config
from typing import Tuple
import random

class DataLoader:
    """ Interface data loader for loading the ImageNet dataset with specified augmentations. """
    def __init__(self, batch_size: int, target_size: Tuple[int]) -> None:
        # Paths relative to working directory
        self.base_path = config['image_net_path']
        self.train_path = config['train_path']
        self.test_path = config['test_path']

        # Configs
        self.batch_size = batch_size
        self.target_size = target_size
        self.seed = random.randint(0, 1000)

    def load_train_set(self, aug_train: CustomDataGenerator, class_mode='sparse', shuffle=True) -> CustomIterator:
        """ Loads the training set from the ImageNet directory using the passed ImageDataGenerator transformations and specified sub-sampling.

        Args:
            aug_train (CustomDataGenerator): Transformations and sub-sampling should be set ahead of time and passed through this parameter.
            class_mode (str, optional): Class mode for labels. Defaults to 'sparse'.
            shuffle (bool, optional): Enable shuffling. Defaults to True.

        Returns:
            CustomIterator: TensorFlow directory iterator.
        """
        print('Loading train set...')
        train_generator = aug_train.flow_from_directory(
                os.path.join(self.base_path, self.train_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode=class_mode,
                seed=self.seed,
                subset='training',
                shuffle=shuffle)

        return train_generator

    def load_val_set(self, aug_val: CustomDataGenerator, class_mode='sparse', shuffle=True) -> CustomIterator:
        """ Loads the validation set from the ImageNet directory using the passed ImageDataGenerator.

        Args:
            aug_val (CustomDataGenerator): TensorFlow ImageDataGenerator.
            class_mode (str, optional): Class mode for labels. Defaults to 'sparse'.
            shuffle (bool, optional): Enable shuffling. Defaults to True.

        Returns:
            CustomIterator: TensorFlow directory iterator.
        """
        print('Loading validation set...')
        val_generator = aug_val.flow_from_directory(
                os.path.join(self.base_path, self.train_path),
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode=class_mode,
                seed=self.seed,
                subset='validation',
                shuffle=shuffle)

        return val_generator

    def load_test_set(self, aug_test: CustomDataGenerator, class_mode=None, shuffle=False, set_batch_size=False):
        """ Loads the test set from the ImageNet directory using the passed ImageDataGenerator.

        Args:
            aug_test (CustomDataGenerator): TensorFlow ImageDataGenerator which should have no sub-sampling or transformations.
            class_mode (str, optional): Class mode for labels. Defaults to None.
            shuffle (bool, optional): Enable shuffling. Defaults to False.
            set_batch_size (bool, optional): Toggle using loader specified batch size or batch size 1. Defaults to False.
        """
        print('Loading test set...')
        test_generator = aug_test.flow_from_directory(
                os.path.join(self.base_path, self.test_path),
                target_size=self.target_size,
                batch_size=1 if not set_batch_size else self.batch_size,
                class_mode=class_mode,
                shuffle=shuffle)
        
        return test_generator