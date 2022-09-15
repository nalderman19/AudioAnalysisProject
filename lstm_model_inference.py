import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"

def load_data(data_path):
    '''
    Returns two numpy arrays containing input data and targets given a dataset.
        
    Parameters
    ----------
    dataset_path (str): Filepath of dataset.
    
    Returns
    -------
    X (ndarray): Input data matrix.
    y (ndarray): Input data targets matrix.
    '''
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data['mfcc'])
    y = np.array(data['labels'])
    
    return X, y


def prepare_dataset(test_size, validation_size):
    '''
    Splits data into train, test and validation datasets.
    
    Parameters
    ----------
    test_size : (float)
        Float indicating the percentage of the dataset to be reserved for testing.
    validation_size : (float)
        Float indicating the percentage of the dataset to be reserved for testing.

    Returns
    -------
    X_train, X_validation, X_test, y_train, y_validation, y_test (np.array): Train/Validation/Test split data.

    '''
    
    X, y = load_data(DATASET_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def predict(model, X, y):
    '''
    Uses model passed in to make a prediction given a datapoint
    
    Parameters
    ----------
    model : ()
        Keras Model
    X : (ndarray)
        Input data
    y : (int)
        Target data
    
    Returns
    -------
    None.
    '''    
    
    # prediction is a 2d array
    #prediction = model.predict(X)
    prediction = model.predict(X[np.newaxis, ...])
    
    # pick out max probability out of 10 genres
    predicted_index = np.argmax(prediction, axis=1)
    
    print("Expected index: {}, Predicted Index: {}".format(y, predicted_index))


if __name__ == "__main__":
    
    # get train, test, validation splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)
    
    # load model
    print("Loading...")
    model = keras.models.load_model("models/myFirstLSTM")
    print("Loaded!!!!")
    
    # evaluate model (not correct way since the train/test split from model fitting will be different from the split in this script, just poc)
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: {}".format(test_accuracy))
    
    # get datapoint to predict
    X = X_test[10]
    y = y_test[10]
    
    # get prediction
    predict(model, X, y)
    
    