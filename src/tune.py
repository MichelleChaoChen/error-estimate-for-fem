import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

train_features, train_labels, test_features, test_labels = None


def model_builder(hp):
    model = keras.Sequential()

    feature_normalizer = layers.Normalization(input_shape=train_features.shape, axis=-1)
    feature_normalizer.adapt(train_features)

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_activation = hp.Choice("activation", ["relu", "tanh"])
    model.add(feature_normalizer)
    model.add(layers.Dense(units=hp_units, activation=hp_activation))
    model.add(layers.Dense(units=hp_units, activation=hp_activation))
    model.add(layers.Dense(units=hp_units, activation=hp_activation))
    model.add(layers.Dense(1))

    # Tune whether to use dropout
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['mse']
    )

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)
