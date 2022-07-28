#AWS Sagemaker tools

import boto3


class Boto:
    bucket_name = 'machine-learning'
    s3_resource = boto3.resource('s3')
    s3_client = boto3.client('s3')

    s3_bucket = s3_resource.Bucket(bucket_name)