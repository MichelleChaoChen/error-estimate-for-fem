import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import keras_tuner as kt
from keras import layers
from functools import partial


class NeuralNetwork:
    def __init__(self, dataset, epochs):
        """
        Initializes the neural network with data and a tuner.

        :param dataset: Dataset for training and testing
        :param epochs: The number of epochs for which it should be trained
        """
        # Split dataframe to features and labels
        self.features, self.labels = self.get_attr(dataset)
        # Split data between training and testing
        self.train_features, self.test_features, self.train_labels, \
            self.test_labels = train_test_split(self.features, self.labels, test_size=0.2)
        # Build tuner for the neural network
        self.tuner = kt.Hyperband(self.build_model,
                                  objective='val_loss',
                                  max_epochs=10,
                                  factor=3,
                                  directory='hyperparameters',
                                  project_name='error_estimator')
        print("START TUNING")
        best_hps = self.tune_model()
        # best_hps = None
        print("START TRAINING")
        self.model = self.train(best_hps, epochs)
        print("START EVALUATING")
        self.evaluate()

    def get_attr(self, data):
        labels = np.array(data.pop(data.shape[1] - 1))
        features = np.array(data)

        return features, labels

    def custom_loss(self, y_true, y_pred):
        """
        Used for constructing/testing different loss functions. 

        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Loss of current predictions
        """
        loss = tf.norm((y_true - y_pred) / y_true)
        return loss

    def build_model(self, hp):
        """
        Builds a model for tuning hyperparameters and for training if 
        the best hyperparameters have already been determined. 

        :param hp: Hyperparameters
        :return: Neural network
        """
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
                    activation=hp.Choice("activation", ["relu"])
                )
            )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        model.add(layers.Dense(1))
        lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        )
        return model

    def train(self, best_hps, epochs=100):
        """
        Trains the model using the best parameters. 

        :param best_hps: Best hyperparameters
        :param epochs: Number of epochs to train for, defaults to 100
        :return: Trained neural network
        """
        model = self.tuner.hypermodel.build(best_hps)
        model.fit(self.train_features, self.train_labels, epochs=epochs,
                  validation_split=0.2, verbose=2)
        return model
        
    def tune_model(self):
        """
        Tunes the neural network by searching 
        for the best hyperparameters. 

        :return: The best hyperparameters found
        """
        self.build_model(kt.HyperParameters())
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.tuner.search(self.train_features, self.train_labels,
                          validation_split=0.2, callbacks=[stop_early])
        best_hps = self.tuner.get_best_hyperparameters(1)[0]
        best_model = self.tuner.get_best_models(num_models=2)[0]
        best_model.summary()
        return best_hps

    def plot_training_loss(self, history):
        """
        Plots how the loss evolves as the model is trained. 
        This is not called by default, add the method call to the constructor 
        if a visualisation of training is needed. 

        :param history: Training history
        """
        epochs = range(1, history.params['epochs'] + 1)
        plt.plot(epochs, history.history['loss'], label='Training Loss')
        plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/training-loss.svg')

    def evaluate(self):
        """
        Evaluates the performance of the neural network on testing data. 
        """
        score = self.model.evaluate(self.test_features, self.test_labels, verbose=2)
        print("EVALUATED MSE {}".format(score))

    def save(self, name):
        """
        Saves the model in h5 format in the models directory.

        :param name: The name the model should be saved as 
        """
        if self.model is not None:
            self.model.save("models/" + name + ".h5")


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
