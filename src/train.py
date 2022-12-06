import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

class NeuralNetwork: 
    def __init__(self, dataset, epochs):
        self.features, self.labels = self.get_attr(dataset)
        self.train_features, self.test_features,\
            self.train_labels, self.test_labels = train_test_split(self.features, self.labels, test_size=0.2)

        print("START TRAINING")
        self.train(epochs)

        print("START EVALUATING")
        self.evaluate()

    def get_attr(self, data):
        labels = np.array(data.pop(data.shape[1] - 1))
        features = np.array(data)

        return features, labels

    def train(self, epochs=100):
        feature_normalizer = layers.Normalization(
            input_shape=np.array(self.train_features[0]).shape,
            axis=-1
        )
        feature_normalizer.adapt(np.array(self.train_features))

        self.model = keras.Sequential([
            feature_normalizer,
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        self.model.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam(0.001))

        # train the model and plot results
        history = self.model.fit(self.train_features, self.train_labels, epochs=epochs,
                                 validation_split=0.2, verbose=2)
        self.plot_training_loss(history)

    def plot_training_loss(self, history):
        epochs = range(1, history.params['epochs'] + 1)
        plt.plot(epochs, history.history['loss'], label='Training Loss')
        plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training-loss.svg')

    def evaluate(self):
        score = self.model.evaluate(self.test_features, self.test_labels, verbose=2)
        print("EVALUATED MSE {}".format(score))

    def save(self, name):
        if self.model is not None:
            self.model.save("models/"+name+".ckpt")


def main(args):
    data = pd.read_csv(args.data, sep=",", header=None)

    error_estimate_model = NeuralNetwork(data, int(args.epochs))
    error_estimate_model.save(args.data.split("/")[-1].split(".")[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data")

    parser.add_argument('-epochs',
                        '--epochs',
                        required=False,
                        default=100,
                        help="number of epochs for training")

    args = parser.parse_args()
    main(args)
