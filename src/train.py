import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner as kt
from keras import layers


class NeuralNetwork:
    def __init__(self, dataset, epochs):
        self.features, self.labels = self.get_attr(dataset)
        self.train_features, self.test_features, \
            self.train_labels, self.test_labels = train_test_split(self.features, self.labels, test_size=0.2)

        print("START TRAINING")
        self.model = self.train(epochs)
        # self.tune_model()
        print("START EVALUATING")
        self.evaluate()

    def get_attr(self, data):
        labels = np.array(data.pop(data.shape[1] - 1))
        features = np.array(data)

        return features, labels

    def build_model(self, hp):
        model = keras.Sequential()
        feature_normalizer = layers.Normalization(
            input_shape=self.train_features[0].shape,
            axis=-1
        )
        feature_normalizer.adapt(self.train_features)
        model.add(feature_normalizer)
        for i in range(hp.Int("num_layers", min_value=1, max_value=10)):
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512),
                    activation=hp.Choice("activation", ["relu", "tanh"])
                )
            )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(1))
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        )
        return model

    def train(self, epochs=100):
        feature_normalizer = layers.Normalization(
            input_shape=self.train_features[0].shape,
            axis=-1
        )
        feature_normalizer.adapt(self.train_features)
        model = keras.Sequential()
        model.add(feature_normalizer)
        model.add(layers.Dense(units=215, activation='relu'))
        model.add(layers.Dense(units=507, activation='relu'))
        model.add(layers.Dense(units=243, activation='relu'))
        model.add(layers.Dense(units=74, activation='relu'))
        model.add(layers.Dense(units=159, activation='relu'))
        model.add(layers.Dense(units=402, activation='relu'))
        model.add(layers.Dense(units=1))
        lr = 0.0037830047995581683
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        )
        model.fit(self.train_features, self.train_labels, epochs=epochs,
                  validation_split=0.2, verbose=2)
        return model

    def tune_model(self):
        self.build_model(kt.HyperParameters())
        tuner = kt.Hyperband(self.build_model,
                             objective='val_loss',
                             max_epochs=10,
                             factor=3,
                             directory='model',
                             project_name='hyperparams')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(self.train_features, self.train_labels, validation_split=0.2, callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(1)[0]
        best_model = tuner.get_best_models(num_models=2)[0]
        best_model.summary()
        print(best_hps.values)
        return best_hps

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
            self.model.save("models/" + name + ".ckpt")


def main(args):
    data = pd.read_csv(args.data, sep=",", header=None)

    error_estimate_model = NeuralNetwork(data, int(args.epochs))
    error_estimate_model.save(os.path.basename(args.data).split('.')[0])


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
