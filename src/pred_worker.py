"""
Worker function for pulling model and test data from S3, loading model, doing
prediction and saving the predictions to S3.
"""

import sys
import os
import jsonref
import pandas as pd

import tensorflow as tf

from modules.botoHelpers import upload_file, download_all_objs_in_folder

config = jsonref.load(open("../config/config-pred.json"))
config_data = config["data"]
config_model = config["model"]
config_pred = config["predictions"]

# ----------


def main():

    try:
        s3_bucket = os.environ["S3BUCKET"]
        s3_bucket_data_folder = os.environ["S3BUCKET_DATA_FOLDER"]
        s3_bucket_savedmodels_folder = os.environ["S3BUCKET_SAVEDMODELS_FOLDER"]
        s3_bucket_pred_folder = os.environ["S3BUCKET_PRED_FOLDER"]

    except KeyError:
        print(
            (
                "Please set the environment variables 'S3BUCKET', "
                "'S3BUCKET_DATA_FOLDER', 'S3BUCKET_SAVEDMODELS_FOLDER' "
                " and 'S3BUCKET_PRED_FOLDER'."
            )
        )
        sys.exit(1)

    # ====================
    #  DOWNLOAD TEST DATA
    # ====================
    _ = download_all_objs_in_folder(
        bucket=s3_bucket,
        bucket_folder=s3_bucket_data_folder,
        target_dir=config_data["data_dir"],
    )

    test_csv_path = os.path.join(
        config_data["data_dir"], config_data["test_csv_filename"]
    )
    test_df = pd.read_csv(test_csv_path, header=0)

    # ======================
    #  DOWNLOAD SAVED MODEL
    # ======================
    _ = download_all_objs_in_folder(
        bucket=s3_bucket,
        bucket_folder=s3_bucket_savedmodels_folder,
        target_dir=config_model["saved_models_dir"],
    )

    saved_model_path = os.path.join(
        config_model["saved_models_dir"], config_model["model_name"]
    )
    model = tf.keras.models.load_model(saved_model_path)

    # =========
    #  PREDICT
    # =========
    test_arr = test_df["x"]
    pred = model.predict(test_arr)
    test_df["y_pred"] = pred

    # ==========================
    #  SAVE PREDICTIONS LOCALLY
    # ==========================
    preds_csv_path = os.path.join(
        config_pred["predictions_dir"],
        config_pred["predictions_csv_filename"],
    )
    test_df.to_csv(preds_csv_path, index=False)

    # ==========================
    #  UPLOAD PREDICTIONS TO S3
    # ==========================
    upload_file(
        file_name=preds_csv_path,
        bucket=s3_bucket,
        bucket_folder=s3_bucket_pred_folder,
    )


# ----------

if __name__ == "__main__":
    main()
