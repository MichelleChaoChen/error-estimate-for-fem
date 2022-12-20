import keras
import numpy as np


def get_error_estimate(features):
    neural_network = keras.models.load_model("models/12_features_scaled_100.ckpt")
    local_errors = neural_network.predict(features)
    global_error = np.linalg.norm(local_errors)
    return local_errors, global_error

