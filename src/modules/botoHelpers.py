import os
import jsonref
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


def download_all_objs_in_folder(bucket, bucket_folder, target_dir):

    """
    Download all files from a specified folder in an S3 bucket.

    Parameters
    ----------
    bucket : str
        Bucket to upload to
    bucket_folder : str
        The folder in the bucket to upload to. E.g. /data/data_subfolder
    target_dir : str
        The target local directory to download the files into.

    Returns
    -------
    List of paths to downloaded files (i.e. paths to local files).
    """

    s3_resource = boto3.resource("s3")

    my_bucket = s3_resource.Bucket(bucket)

    objects = my_bucket.objects.filter(Prefix=bucket_folder)

    try:

        files_downloaded = []

        for obj in objects:
            # Skip the obj that is the bucket_folder itself.
            if obj.key == "data/":
                continue
            # Don't skip all the other objs with obj.key = "bucket_folder/...".
            else:
                path, filename = os.path.split(obj.key)
                download_to_file_path = os.path.join(target_dir, filename)
                my_bucket.download_file(Key=obj.key, Filename=download_to_file_path)
                files_downloaded.append(download_to_file_path)

    except ClientError as e:
        return False

    return files_downloaded


# def main():

#     s3_client = boto3.client("s3")
#     upload_file(file_name="../../output/predictions.csv", bucket="cleon-docker-test")


# if __name__ == "__main__":
#     main()
