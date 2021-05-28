"""
Worker function for pulling data from S3, creating the TF data set, building
the TF model, training the model and saving the trained model to S3.
"""

import sys
import os
import jsonref
import pandas as pd

from modules.botoHelpers import upload_file, download_all_objs_in_folder
from modules.trainHelpers import dataframe_to_dataset, build_model, train_model

config = jsonref.load(open("../config/config-train.json"))
_rseed = config["rseed"]
config_data = config["data"]
config_model = config["model"]

# ----------


def main():

    try:
        s3_bucket = os.environ["S3BUCKET"]
        s3_bucket_data_folder = os.environ["S3BUCKET_DATA_FOLDER"]
        s3_bucket_savedmodels_folder = os.environ["S3BUCKET_SAVEDMODELS_FOLDER"]
    except KeyError:
        print(
            (
                "Please set the environment variables 'S3BUCKET',"
                "'S3BUCKET_DATA_FOLDER'. and 'S3BUCKET_SAVEDMODELS_FOLDER'."
            )
        )
        sys.exit(1)

    # ===================
    #  Pull data from S3
    # ===================
    # Download data/train.csv and data/test.csv from S3 bucket into local ../data folder.
    _ = download_all_objs_in_folder(
        bucket=s3_bucket,
        bucket_folder=s3_bucket_data_folder,
        target_dir=config_data["data_dir"],
    )

    # ====================================
    #  Train-val split, create TF dataset
    # ====================================

    train_csv_path = os.path.join(
        config_data["data_dir"], config_data["train_csv_filename"]
    )

    # Load csv.
    df = pd.read_csv(train_csv_path, header=0)

    # Train test split.
    train_df = df.sample(frac=config_data["train_split"], random_state=_rseed)
    val_df = df.drop(train_df.index)

    # Create tf dataset.
    train_ds = dataframe_to_dataset(df=train_df)
    val_ds = dataframe_to_dataset(df=val_df)

    # Batch the dataset.
    train_ds = train_ds.batch(config_model["batch_size"])
    val_ds = val_ds.batch(config_model["batch_size"])

    # for x, y in train_ds.take(1):
    #     print("Input:", x)
    #     print("Target:", y)

    # =============
    #  Build model
    # =============

    model = build_model(intermediate_neurons=config_model["intermediate_neurons"])
    print(model.summary())

    # =============
    #  Train model
    # =============

    train_model(
        model=model,
        learning_rate=config_model["learning_rate"],
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=config_model["epochs"],
    )

    # ====================
    #  Save model locally
    # ====================

    saved_model_path = os.path.join(
        config_model["saved_models_dir"], config_model["model_name"]
    )
    model.save(saved_model_path)

    # ================================
    #  Push locally saved model to S3
    # ================================

    upload_file(
        file_name=saved_model_path,
        bucket=s3_bucket,
        bucket_folder=s3_bucket_savedmodels_folder,
    )

    return


# ----------

if __name__ == "__main__":
    main()
