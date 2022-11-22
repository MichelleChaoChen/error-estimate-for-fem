import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from show_progress import Show_Progress

class NeuralNetwork: 
    def __init__(self, train_data):
        self.train_labels = np.array(train_data.pop(train_data.shape[1] - 1))
        self.train_features = np.array(train_data)

        self.train()
    
    def train(self):
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

        self.model.compile(loss='mean_absolute_error', 
            optimizer=tf.keras.optimizers.Adam(0.001))

        # train the model 
        self.model.fit(self.train_features, 
            self.train_labels, 
            epochs=100, 
            verbose=0, 
            validation_split=0.2,
            callbacks=[
                Show_Progress(),
            ]
        )

    def save(self, name):
        self.model.save("models/"+name+".ckpt")

def main(args):
    data = pd.read_csv(args.data, sep=" ", header=None)

    error_estimate_model = NeuralNetwork(data)
    error_estimate_model.save(args.data.split("/")[-1].split(".")[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-data',
                        '--data',
                        required=True,
                        help="path to training data")

    args = parser.parse_args()
    main(args)
