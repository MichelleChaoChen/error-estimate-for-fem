import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from show_progress import Show_Progress

# path to checkpoint model.
checkpoint_path = "models/cp.ckpt"

class Neural_NetWork:
    def __init__(self):
        self.model = None
        pass

    def load_model(self, model_path = checkpoint_path):
        self.model = keras.models.load_model(model_path)
        pass

    def predict(self, features):
        if self.model is None:
            print("NOT FOUND MODEL")
            return 

        return self.model.predict(features)

    def train(self, train_features, train_labels):
        train_features, train_labels = np.array(train_features), np.array(train_labels)
    
        feature_normalizer = layers.Normalization(
            input_shape=np.array(train_features[0]).shape, 
            axis=-1
        )

        feature_normalizer.adapt(np.array(train_features))
        
        self.model = keras.Sequential([
            feature_normalizer, 
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        self.model.compile(loss='mean_absolute_error', 
            optimizer=tf.keras.optimizers.Adam(0.001))

        self.model.fit(train_features, 
            train_labels, 
            epochs=100, 
            verbose=0, 
            validation_split=0.2,
            callbacks=[
                Show_Progress(),
                tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weight_only=True, verbose=1)
            ]
        )
        pass
