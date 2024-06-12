import os
import boto3
from botocore.client import Config

def download_files(download_path: str):
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    # Create download path if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Create an S3 client with MinIO configuration
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version='s3v4')
    )
    
    # List all the objects in the bucket and filter out the PDFs
    response = s3.list_objects_v2(Bucket=bucket_name)
    pdf_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pdf')]

    # Download files
    for key in pdf_keys:
        file_name = os.path.basename(key)
        local_file_path = os.path.join(download_path, file_name)
        s3.download_file(bucket_name, key, local_file_path)

        # Rename the file in S3 to mark it as done
        s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': key}, Key=f"{key}.done")
        s3.delete_object(Bucket=bucket_name, Key=key)
