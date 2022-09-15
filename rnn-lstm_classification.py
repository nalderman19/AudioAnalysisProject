import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os

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


def build_model(input_shape):
    '''
    Generates RNN-LSTM Model 
    
    Parameters
    ----------
    input_shape : (tuple)
        Tuple containing the shape of the input set
        
    Returns
    -------
    model : ()
        Keras Model - RNN-LSTM
        
    '''
    
    model = keras.Sequential()
    
    # 1st lstm layer
    model.add(keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    
    # 2nd lstm layer
    model.add(keras.layers.LSTM(64))

    # 1st dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    
    # output softmax layer
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    return model


def plot_history(history):
    '''
    Uses matplotlib to plot accuracy/ loss for training and validatoin set as a function of epochs.
    
    Parameters
    ----------
    history : ()
        Training history of the model.
    
    Returns
    -------
    None.
    '''
    
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    
    # get train, test, validation splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)
    
    # create model
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)
    
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    
    # train model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
    
    # plot accuracy/ loss for training and validation
    plot_history(history)
    
    # evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    # save model
    model.save("models/myFirstLSTM")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    