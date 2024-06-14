from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath, component, pipeline
from kfp import kubernetes
# from download_files import download_files as download_files_func
# from load_files import load_files as load_files_func
    
def download_files(download_path: str):
    import os
    import boto3
    from botocore.client import Config

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
        # s3.delete_object(Bucket=bucket_name, Key=key)    

@component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "langchain", "langchain-community"]
)
def load_files(download_path: InputPath(str)):
    import os
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Milvus

    download_files(download_path=download_path)

    # Process the downloaded PDFs
    pdfs_to_urls = {}
    for file_name in os.listdir(download_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(download_path, file_name)
            print(f"Processing file: {file_path}")
            # Remove extension from file name
            file_name = file_name.replace('.pdf', '')
            # Add the url of the PDF in the S3 bucket to pdfs_to_urls dictionary
            pdfs_to_urls[file_name] = file_path
        
    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(download_path)
    pdf_docs = pdf_loader.load()

    # Inject metadata
    from pathlib import Path

    for doc in pdf_docs:
        doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).stem]

    # Split documents into chunks with some overlap
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(pdf_docs)

    # Create the index and ingest the documents
    model_kwargs = {'trust_remote_code': True, 'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs=model_kwargs,
        show_progress=True
    )

    milvus_host = os.getenv('MILVUS_HOST')
    milvus_port = os.getenv('MILVUS_PORT')
    milvus_username = os.getenv('MILVUS_USERNAME')
    milvus_password = os.getenv('MILVUS_PASSWORD')
    milvus_collection = os.getenv('MILVUS_COLLECTION_NAME')

    db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_username, "password": milvus_password},
        collection_name=milvus_collection,
        metadata_field="metadata",
        text_field="page_content",
        auto_id=True,
        drop_old=True
    )
    
    db.add_documents(all_splits)

    print("done")

@pipeline(
    name='Download and Load PDFs from MinIO Pipeline',
    description='A pipeline to download and load PDFs from a MinIO instance.'
)
def download_and_load_pipeline():
    load_task = load_files(download_path='/mnt/data')
        
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
