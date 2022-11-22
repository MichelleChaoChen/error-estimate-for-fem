from neural_network import Neural_NetWork
import numpy as np

def estimate(features):
    error_estimator = Neural_NetWork()
    error_estimator.load_model()

    return error_estimator.predict(features)


