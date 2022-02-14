import numpy as np

def test_model(model, test_set) -> None:
    print('\nPredicting on test-set...')
    test_set.reset()
    pred = model.predict(test_set, steps=test_set.n//test_set.batch_size, verbose=1)
    pred = np.argmax(pred, axis=-1)
    cls = np.array(test_set.classes)
    
    print('Computing accuracy...')
    accuracy = np.sum(pred == cls)/len(pred)
    print(f'Accuracy: {accuracy}')