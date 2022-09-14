import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    
    return X, y

def prepare_datasets(test_size, validation_size):
    # load in data
    X, y = load_data(DATASET_PATH)
    # create train/ test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # create train/ validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # cnn tensorflow expects a 3d array
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # create model
    model = keras.Sequential()
    
    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization()) # process that normalizes the activations that get sent to next layer and speeds up training and improves model
    
    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu"))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu"))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))
    model.add(keras.layers.BatchNormalization())
    
    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    # output softmax layer
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    return model

    
def predict(model, X, y):
    
    # prediction is a 2d array
    prediction = model.predict(X[np.newaxis, ...])
    
    # pick out max probabbility out of 10 genres
    predicted_index = np.argmax(prediction, axis=1)
    
    print("Expected index: {}, Predicted Index: {}".format(y, predicted_index))
    


if __name__ == "__main__":
    # create train, validation, test datasets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25,0.2)
    
    # build cnn net
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # train the cnn
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)
    
    # evaluate the cnn on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: {}".format(test_accuracy))
    
    # make prediction on a sample
    X = X_test[100]
    y = y_test[100]
    predict(model, X, y)