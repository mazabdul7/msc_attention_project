import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

class CustomIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator, target_size=..., color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, data_format=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest', dtype=None):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, dtype)
        self.num_classes = len(self.class_indices)
        self.weights = None
        self.target_classes = None
        self.cls_to_file_idx = None

    def set_subsample(self, target_classes, target_weights):
        ''' Takes args for fixed target classes and their desired spread amongst other classes. '''
        assert sum(target_weights) <= 1
        classes = [self.class_indices[t_class] for t_class in target_classes]

        for cl in range(self.num_classes):
            self.cls_to_file_idx[cl] = list(np.nonzero(np.array(self.classes)==cl)[0])

        self.weights = target_weights
        self.target_classes = classes
    
    def reset_subsample(self):
        self.target_classes = None
        self.weights = None

    def _set_index_array(self):
        if not self.target_classes:
            self.index_array = np.arange(self.n)
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

        if self.shuffle:
            np.random.shuffle(self.index_array)

class CustomDataGenerator(ImageDataGenerator):
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