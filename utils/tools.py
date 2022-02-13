import numpy as np

def test_model(model, test_set) -> None:
    print('Predicting on test-set...')
    test_set.reset()
    pred = model.predict(test_set, steps=test_set.n//test_set.batch_size, verbose=1)
    pred = np.argmax(pred, axis=1)
    idx_to_cls = {v: k for k, v in test_set.class_indices.items()} # Convert prediction to unique file handlers
    preds_cls = np.vectorize(idx_to_cls.get)(pred)
    filenames = [name.split('/')[-2] for name in test_set.filenames] # Get class labels from folder names
    
    print('Computing accuracy...')
    accuracy = sum([preds_cls[i] == filenames[i] for i in range(len(preds_cls))])/test_set.n
    print(f'Accuracy: {accuracy}')