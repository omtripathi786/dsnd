# Importing pandas and numpy
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils


def one_hot_encoding(data):
    # Make dummy variables for rank
    one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
    # Drop the previous rank column
    one_hot_data = one_hot_data.drop('rank', axis=1)
    # Print the first 10 rows of our data
    return one_hot_data[:]


def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(6,)))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(2, activation='softmax'))
    return model


if __name__ == '__main__':
    data = pd.read_csv('CSV_DATA/sd.csv')
    # plot_points(data)
    processed_data = one_hot_encoding(data)
    sample = np.random.choice(processed_data.index, size=int(len(processed_data) * 0.9), replace=False)
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

    # Splitting the data into features and targets (labels)
    """
    Now, as a final step before the training, we'll split the data into features (X) and targets (y).
    Also, in Keras, we need to one-hot encode the output. We'll do this with the to_categorical function.
    Separate data and one-hot encode the output.
    Note: We're also turning the data into numpy arrays, in order to train the model in Keras
    """
    features = np.array(train_data.drop('admit', axis=1))
    targets = np.array(keras.utils.to_categorical(train_data['admit'], 2))
    features_test = np.array(test_data.drop('admit', axis=1))
    targets_test = np.array(keras.utils.to_categorical(test_data['admit'], 2))

    model = build_model()
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Training the model
    model.fit(features, targets, epochs=200, batch_size=100, verbose=0)

    # Evaluating the model on the training and testing set
    score = model.evaluate(features, targets)
    print("\n Training Accuracy:", score[1])
    score = model.evaluate(features_test, targets_test)
    print("\n Testing Accuracy:", score[1])
