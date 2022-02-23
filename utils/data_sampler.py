import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from typing import List

class CustomIterator(DirectoryIterator):
    """ Custom TensorFlow Directory Iterator that implements sub-sampling at the class level. """
    def __init__(self, directory, image_data_generator, target_size=..., color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest', dtype=None):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, dtype)
        self.num_classes = len(self.class_indices)
        self.weights = None
        self.target_classes = None
        self.subsample_size = None
        self.old_n = int(self.n)
        self.cls_to_file_idx = dict()

    def set_target_sampling(self, target_classes: List[str], target_weights: List[int]) -> None:
        """ Enables and sets targetted sub-sampling weights for the target classes.

        Args:
            target_classes (List[str]): The target class folder name to fix sampling weights to.
            target_weights (List[int]): The fraction amongst which each target class comprises the final ImageNet dataset.

        Example:
            target_classes = ['n03873416'], target_weights = [5/1000]
            The target class 'n03873416' will then comprise 5/1000 of the ImageNet dataset with the remaining 995/1000 being
            uniformly randomly sampled across all remaining non-target classes. 
            Note: The largest specified weighted target class is loaded in its ENTIRETY, and therefore determines the final dataset size.
        """
        self.reset_subsampling()
        assert sum(target_weights) <= 1
        classes = [self.class_indices[t_class] for t_class in target_classes] # Converts string folder name class to actual inner class mapping

        # Mapping between class and its indices in the dataset
        for cl in range(self.num_classes):
            self.cls_to_file_idx[cl] = list(np.nonzero(np.array(self.classes)==cl)[0])

        self.weights = target_weights
        self.target_classes = classes
        self._set_index_array()

    def set_subsampling(self, size: int) -> None:
        """ Set subsampling with no target classes.

        Args:
            size (int): Size of ImageNet sample.
        """
        self.reset_target_sampling()
        self.subsample_size = size
        self._set_index_array()

    def reset_subsampling(self) -> None:
        """ Disable sub-sampling and reset dataset size.
        """
        self.subsample_size = None
        self.n = self.old_n
        self._set_index_array()
    
    def reset_target_sampling(self) -> None:
        """ Disable target class sub-sampling and reset dataset size.
        """
        self.target_classes = None
        self.weights = None
        self.n = self.old_n
        self._set_index_array()

    def _set_index_array(self):
        """ Generates the index array according to the set sub-sampling rates. If sub-sampling is disabled uses entire default ImageNet.
            Note: This function is naturally called at the end of each epoch through model.fit() - must be called to resample!
        """
        if not self.target_classes:
            self.index_array = np.arange(self.n)
            if self.subsample_size:
                sample = np.random.choice(self.index_array, replace=False, size=self.subsample_size)
                self.index_array = sample
                self.n = int(self.subsample_size)
        else:
            self.index_array = []
            max_cl = np.argmax(self.weights)
            max_size = int((1/self.weights[max_cl]) * len(self.cls_to_file_idx[self.target_classes[max_cl]]))
            self.index_array.extend(self.cls_to_file_idx[self.target_classes[max_cl]])
            for i, cl in enumerate(self.target_classes):
                if cl != self.target_classes[max_cl]:
                    sample = np.random.choice(self.cls_to_file_idx[cl], replace=False, size=int(self.weights[i]*max_size))
                    self.index_array.extend(sample)
            remainder = max_size - len(self.index_array)
            if remainder != 0:
                remaining_classes = []
                for cl in self.cls_to_file_idx.keys():
                    if cl not in self.target_classes:
                        remaining_classes.extend(self.cls_to_file_idx[cl])
                sample = np.random.choice(remaining_classes, replace=False, size=int(remainder))
                self.index_array.extend(sample)
            self.n = int(max_size)

        if self.shuffle:
            np.random.shuffle(self.index_array)

class CustomDataGenerator(ImageDataGenerator):
    """ Custom ImageDataGenerator that implements the above sub-sampling directory iterator with specified data augmentation. """
    def __init__(self, featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=0.000001, rotation_range=0, width_shift_range=0, height_shift_range=0, brightness_range=None, shear_range=0, zoom_range=0, channel_shift_range=0, fill_mode='nearest', cval=0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0, dtype=None):
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split, dtype)

    def flow_from_directory(self, directory, target_size=..., color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest'):
        return CustomIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)