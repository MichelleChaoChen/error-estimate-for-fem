import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

class NeuralNetwork: 
    def __init__(self, dataset, epochs):
        train_data, test_data = self.split_data(dataset, 0.2) # 0.8 for training, 0.2 for testing
        self.train_features, self.train_labels = self.get_attr(train_data)
        self.test_features, self.test_labels = self.get_attr(test_data)

        print("START TRAINING")
        self.train(epochs)

        print("START EVALUATING")
        self.evaluate()

    def get_attr(self, data):
        labels = np.array(data.pop(data.shape[1] - 1))
        features = np.array(data)

        return features, labels

    def split_data(self, dataset, split_ratio):
        test_indices = np.random.rand(len(dataset)) < split_ratio
        return dataset[~test_indices], dataset[test_indices]

    def train(self, epochs=100):
        feature_normalizer = layers.Normalization(
            input_shape=np.array(self.train_features[0]).shape, 
            axis=-1
        )

        feature_normalizer.adapt(np.array(self.train_features))
        
        self.model = keras.Sequential([
            feature_normalizer, 
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        self.model.compile(loss='mean_squared_error', 
            optimizer=tf.keras.optimizers.Adam(0.001))

        # train the model 
        self.model.fit(self.train_features, 
            self.train_labels, 
            epochs=epochs, 
            verbose=2, 
        )

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
