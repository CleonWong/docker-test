"""
Helper functions for `train_worker.py`.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------


def dataframe_to_dataset(df):

    df = df.copy()
    labels = df.pop("y")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))

    return ds


def build_model(intermediate_neurons=10):

    """
    Builds a 5-layer vanilla neural network, with the 3 intermediate layers each
    having `intermediate_neurons` number of neurons.

    1 input neuron
          |
    `intermediate_neurons`
          |
    `intermediate_neurons`
          |
    `intermediate_neurons`
          |
    1 output neuron

    Parameters
    ----------
    intermediate_neurons : int, default=10
        The number of intermediate neurons. Defaults to 10.

    Returns
    -------
    model
    """

    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(1,)))
    # model.add(layers.Dense(1, activation="relu", name="input_neuron"))
    model.add(
        layers.Dense(intermediate_neurons, activation="relu", name="inter_layer_1")
    )
    model.add(
        layers.Dense(intermediate_neurons, activation="relu", name="inter_layer_2")
    )
    model.add(
        layers.Dense(intermediate_neurons, activation="relu", name="inter_layer_3")
    )
    model.add(layers.Dense(1, name="output_neuron"))

    return model


def train_model(model, learning_rate, train_ds, val_ds, epochs):

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
    )
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

    return
