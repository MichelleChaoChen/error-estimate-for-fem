from tensorflow import keras

class Show_Progress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("Loss = {} Validation Loss = {}".format(logs["loss"], logs["val_loss"]))