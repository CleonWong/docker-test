import jsonref
import os
import boto3
from botocore.exceptions import ClientError

# ----------


def upload_file(file_name, bucket, bucket_folder=None, object_name=None):

    """
    Upload a file to an S3 bucket

    Parameters
    ----------
    file_name : str or Path
        File to upload
    bucket : str
        Bucket to upload to
    bucket_folder : str, default=None
        The folder in the bucket to upload to.
    object_name : str, default=None
        S3 object name. If not specified then file_name is used

    Returns
    -------
    ``True`` if file was uploaded, else ``False``.
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        if bucket_folder is None:
            object_name = os.path.basename(file_name)
        else:
            object_name = os.path.join(bucket_folder, os.path.basename(file_name))

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        return False

    return True


# def main():

#     s3_client = boto3.client("s3")
#     upload_file(file_name="../../output/predictions.csv", bucket="cleon-docker-test")


# if __name__ == "__main__":
#     main()
