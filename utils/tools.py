import numpy as np
from tensorflow.keras import Model
from utils.data_sampler import CustomIterator

def test_model(model: Model, test_set: CustomIterator) -> None:
    """ Test the passed model for its Top-1 accuracy on the passed test set.

    Args:
        model (Model): TensorFlow model.
        test_set (CustomIterator): Test set to test on.
    """
    print('\nPredicting on test-set...')
    test_set.reset()
    pred = model.predict(test_set, steps=test_set.n//test_set.batch_size, verbose=1)
    pred = np.argmax(pred, axis=-1)
    cls = np.array(test_set.classes)
    
    print('Computing accuracy...')
    accuracy = np.sum(pred == cls)/len(pred)
    print('\n-----------------------------------------')
    print(f'Model Accuracy on test-set: {accuracy}')
    print('-----------------------------------------\n')