"""
Worker function for generating data points and saving to S3 bucket.
"""

import jsonref
import os
from modules.generateData import generate_data
from modules.botoHelpers import upload_file

config = jsonref.load(open("../config/config-data.json"))

# ----------

def main():

    try:
        s3_bucket = os.environ["S3BUCKET"]
        s3_bucket_folder = os.environ["S3BUCKET_FOLDER"]
    except KeyError:
        print("Please set the environment variable 'S3BUCKET' and 'S3BUCKET_FOLDER.")

    train_csv_path = os.path.join(
        config["data_dir"],
        config["train_csv_filename"],
    )
    test_csv_path = os.path.join(
        config["data_dir"],
        config["test_csv_filename"],
    )

    _, _ = generate_data(
        n_samples=config["train_n_samples"],
        csv_path=train_csv_path,
    )
    _, _ = generate_data(
        n_samples=config["test_n_samples"],
        csv_path=test_csv_path,
    )

    # Upload files to S3 bucket.
    upload_file(
        file_name=train_csv_path,
        bucket=s3_bucket,
        bucket_folder=s3_bucket_folder,
    )
    upload_file(
        file_name=test_csv_path,
        bucket=s3_bucket,
        bucket_folder=s3_bucket_folder,
    )

    return

# ----------

if __name__ == "__main__":
    main()
