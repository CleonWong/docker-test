import os
import jsonref
from datetime import datetime
import pandas as pd

from modules.generateData import generate_data
from modules.botoHelpers import upload_file

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
config = jsonref.load(open("../config/config.json"))
_rseed = config["rseed"]
data_config_dict = config["data"]
model_config_dict = config["model"]
preds_config_dict = config["predictions"]
s3_bucket = config["s3_bucket"]

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
    model.add(keras.Input(shape=(1,)))
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


def train_model(model, lr, train_ds, val_ds, epochs):

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.SGD(lr=lr),
    )
    model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)

    return


# ----------


def main():
    """
    Main function for training neural network.
    """

    os.mkdir(os.path.join(data_config_dict["data_dir"], timestamp_str))
    os.mkdir(os.path.join(preds_config_dict["predictions_dir"], timestamp_str))

    # ======
    #  Data
    # ======

    # Create train and test data.
    train_csv_path = os.path.join(
        data_config_dict["data_dir"],
        timestamp_str,
        data_config_dict["train_csv_filename"],
    )
    test_csv_path = os.path.join(
        data_config_dict["data_dir"],
        timestamp_str,
        data_config_dict["test_csv_filename"],
    )
    _, _ = generate_data(
        n_samples=data_config_dict["train_n_samples"],
        csv_path=train_csv_path,
    )
    _, _ = generate_data(
        n_samples=data_config_dict["test_n_samples"],
        csv_path=test_csv_path,
    )

    # Load csv.
    df = pd.read_csv(train_csv_path, header=0)
    test_df = pd.read_csv(test_csv_path, header=0)

    # Train test split.
    train_df = df.sample(frac=data_config_dict["train_split"], random_state=_rseed)
    val_df = df.drop(train_df.index)

    # Create tf dataset.
    train_ds = dataframe_to_dataset(df=train_df)
    val_ds = dataframe_to_dataset(df=val_df)

    # Batch the dataset.
    train_ds = train_ds.batch(model_config_dict["batch_size"])
    val_ds = val_ds.batch(model_config_dict["batch_size"])

    # for x, y in train_ds.take(1):
    #     print("Input:", x)
    #     print("Target:", y)

    # =======
    #  Model
    # =======

    # Build model.
    model = build_model(intermediate_neurons=model_config_dict["intermediate_neurons"])
    print(model.summary())

    # Train model.
    train_model(
        model=model,
        lr=model_config_dict["learning_rate"],
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=model_config_dict["epochs"],
    )

    # Prediction.
    test_arr = test_df["x"]
    predictions = model.predict(test_arr)
    test_df["y_pred"] = predictions

    # Save outputs locally.
    preds_csv_path = os.path.join(
        preds_config_dict["predictions_dir"],
        timestamp_str,
        preds_config_dict["predictions_csv_filename"],
    )
    test_df.to_csv(preds_csv_path, index=False)

    # ====
    #  S3
    # ====

    # Upload files to S3 bucket.
    upload_file(
        file_name=train_csv_path,
        bucket=s3_bucket,
        bucket_folder=os.path.join("data", timestamp_str),
    )
    upload_file(
        file_name=test_csv_path,
        bucket=s3_bucket,
        bucket_folder=os.path.join("data", timestamp_str),
    )
    upload_file(
        file_name="../config/config.json",
        bucket=s3_bucket,
        bucket_folder=os.path.join("config", timestamp_str),
    )
    upload_file(
        file_name=preds_csv_path,
        bucket=s3_bucket,
        bucket_folder=os.path.join("output", timestamp_str),
    )


if __name__ == "__main__":
    main()
