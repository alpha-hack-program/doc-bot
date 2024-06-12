from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath, component, pipeline
from kfp import kubernetes
from download_files import download_files as download_files_func
from load_files import load_files as load_files_func

@component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3"]
)
def download_files(download_path: OutputPath(str)):
    download_files_func(download_path)

@component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "langchain", "langchain-community"]
)
def load_files(download_path: InputPath(str)):
    load_files_func(download_path)

@pipeline(
    name='Download and Load PDFs from MinIO Pipeline',
    description='A pipeline to download and load PDFs from a MinIO instance.'
)
def download_and_load_pipeline():
    download_task = download_files()
    load_task = load_files(
        download_path=download_task.outputs['download_path']
    )
    
    kubernetes.use_secret_as_env(
        download_task,
        secret_name='aws-connection-documents',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })
    
    kubernetes.use_secret_as_env(
        load_task,
        secret_name='aws-connection-documents',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })
    
    kubernetes.use_secret_as_env(
        load_task,
        secret_name='milvus-connection-documents',
        secret_key_to_env={
            'MILVUS_HOST': 'MILVUS_HOST',
            'MILVUS_PORT': 'MILVUS_PORT',
            'MILVUS_USERNAME': 'MILVUS_USERNAME',
            'MILVUS_PASSWORD': 'MILVUS_PASSWORD',
            'MILVUS_COLLECTION_NAME': 'MILVUS_COLLECTION_NAME'
        })

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=download_and_load_pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )
