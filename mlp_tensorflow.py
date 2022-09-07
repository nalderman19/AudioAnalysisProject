
import numpy as np
from random import random
import tensorflow as tf
from sklearn.model_selection import train_test_split


def generateDataset(num, testSize):
    # made up dataset to aid computing sum of inputs... see structure in spyder variable explorer
    x = np.array([[random() / 2 for _ in range(2)] for _ in range(num)]) # inputs
    y = np.array([[i[0] + i[1]] for i in x])                       # targets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generateDataset(1000,0.2)

    # build model 2 -> [5] -> 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    # compile model
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.3) # Stochastic gradient descent
    model.compile(optimizer=optimiser, loss="MSE") # same loss as manual mlp
    
    
    # train model
    model.fit(x_train, y_train, batch_size=1, epochs=50)
    
    
    # evaluate
    print("\nModel Evaluation:")
    model.evaluate(x_test, y_test, batch_size=1, verbose=1)


    # predictions
    data = np.array([[0.1, 0.2], [0.2,0.2]])
    predictions = model.predict(data, batch_size=1)
    
    