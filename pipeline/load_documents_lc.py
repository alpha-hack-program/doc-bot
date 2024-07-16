# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os
import sys

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from kfp import kubernetes

from kubernetes import client, config
from sympy import im

# This component downloads all the PDFs in an S3 bucket and load them and saves them to the correspoding output paths.
# The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - MODEL_S3_KEY
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore", "langchain-community","pypdf"]
)
def get_chunks_from_documents(
    chunks_output_dataset: Output[Dataset]
):
    import os
    import re
    import pickle

    import boto3
    import botocore
    
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Get the S3 bucket connection details
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    # Connect to the S3 bucket
    session = boto3.session.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name
    )

    bucket = s3_resource.Bucket(bucket_name)

    # Check if the bucket exists, and create it if it doesn't
    if not bucket.creation_date:
        s3_resource.create_bucket(Bucket=bucket_name)

    # Define a temporary directory to store the PDFs
    local_tmp_dir = '/tmp/pdfs'
    print(f"local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Get all objects from the bucket
    objects = bucket.objects.all()
    print(f"objects found: {objects}")

    # Filter and download PDF files
    for obj in objects:
        key = obj.key
        if key.endswith('.pdf'):
            # Define the local file path
            local_file_path = os.path.join(local_tmp_dir, os.path.basename(key))
            
            # Download the file
            bucket.download_file(key, local_file_path)
            print(f'Downloaded {key} to {local_file_path}')

    # Load PDFs from the local directory
    pdf_loader = PyPDFDirectoryLoader(local_tmp_dir)
    pdf_docs = pdf_loader.load()
    print(f"Loaded {len(pdf_docs)} PDF documents")

    # Define a regular expression pattern to match "<ID>-<TYPE>*.pdf"
    pattern = re.compile(r'^' + re.escape(local_tmp_dir + '/') + r'([0-9]+)-(.*)\.pdf$', re.IGNORECASE)
    for doc in pdf_docs:
        match = pattern.match(doc.metadata["source"])
        if match:
            doc.metadata["dossier"] = match.group(1)

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                               chunk_overlap=40)
    chunks = text_splitter.split_documents(pdf_docs)
    print(f"Split {len(pdf_docs)} PDF documents into {len(chunks)} chunks")

    # Save the chunks to the output dataset path
    print(f"Dumping chunks to {chunks_output_dataset.path}")
    with open(chunks_output_dataset.path, "wb") as f:
        pickle.dump(chunks, f)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["langchain-milvus", "langchain-huggingface", "sentence-transformers", "pymilvus", "einops", "openai", "transformers"]
)
def add_chunks_to_milvus(
    chunks_input_dataset: Input[Dataset]
):
    import os
    import pickle
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_milvus import Milvus
    from langchain_core.documents import Document

    # Get the Mivus connection details
    milvus_host = os.environ.get('MILVUS_HOST')
    milvus_port = os.environ.get('MILVUS_PORT')
    milvus_username = os.environ.get('MILVUS_USERNAME')
    milvus_password = os.environ.get('MILVUS_PASSWORD')
    milvus_collection = os.environ.get('MILVUS_COLLECTION')
    
    # Print the connection details
    print(f"milvus_host: {milvus_host}")
    print(f"milvus_port: {milvus_port}")
    print(f"milvus_collection: {milvus_collection}")

    # Get the chunks from the input dataset
     # Try to load the chunks from the input dataset
    try:
        print(f"Loading chunks from {chunks_input_dataset.path}")
        with open(chunks_input_dataset.path, 'rb') as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"Failed to load chunks: {e}")

    # Check if the variable exists and is of the correct type
    if chunks is None:
        raise ValueError("Chunks not loaded successfully.")
    
    if not isinstance(chunks, list) or not all(isinstance(doc, Document) for doc in chunks):
        raise TypeError("The loaded data is not a List[langchain_core.documents.Document].")

    print(f"Loaded {len(chunks)} chunks")

    # If you don't want to use a GPU, you can remove the 'device': 'cuda' argument
    # model_kwargs = {'trust_remote_code': True, 'device': 'cuda'}
    model_kwargs = {'trust_remote_code': True}
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # model_name = "nomic-ai/nomic-embed-text-v1"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_username, "password": milvus_password},
        collection_name=milvus_collection,
        metadata_field="metadata",
        text_field="page_content",
        drop_old=True
        )

# This pipeline will download evaluation data, download the model, test the model and if it performs well, 
# upload the model to the runtime S3 bucket and refresh the runtime deployment.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(accuracy_threshold: float = 0.95,  enable_caching: bool = False):
    # Get all the PDFs from the S3 bucket
    get_chunks_from_documents_task = get_chunks_from_documents().set_caching_options(False)

    # Add chunks to vector store
    add_chunks_to_vector_store_task = add_chunks_to_milvus(
        chunks_input_dataset=get_chunks_from_documents_task.outputs["chunks_output_dataset"]
    ).set_caching_options(False)
        

    # Set the kubernetes secret to be used in the get_chunks_from_documents task
    kubernetes.use_secret_as_env(
        task=get_chunks_from_documents_task,
        secret_name='aws-connection-documents',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

    # Set the kubernetes secret to be used in the add_chunks_to_milvus task
    add_chunks_to_vector_store_task.set_env_variable(name="MILVUS_COLLECTION", value="documents")
    kubernetes.use_secret_as_env(
        task=add_chunks_to_vector_store_task,
        secret_name='milvus-connection-documents',
        secret_key_to_env={
            'MILVUS_HOST': 'MILVUS_HOST',
            'MILVUS_PORT': 'MILVUS_PORT',
            'MILVUS_USERNAME': 'MILVUS_USERNAME',
            'MILVUS_PASSWORD': 'MILVUS_PASSWORD',
        })
    
def get_pipeline_by_name(client: kfp.Client, pipeline_name: str):
    import json

    # Define filter predicates
    filter_spec = json.dumps({
        "predicates": [{
            "key": "display_name",
            "operation": "EQUALS",
            "stringValue": pipeline_name,
        }]
    })

    # List pipelines with the specified filter
    pipelines = client.list_pipelines(filter=filter_spec)

    if not pipelines.pipelines:
        return None
    for pipeline in pipelines.pipelines:
        if pipeline.display_name == pipeline_name:
            return pipeline

    return None

# Get the service account token or return None
def get_token():
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/token", "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Get the route host for the specified route name in the specified namespace
def get_route_host(route_name: str):
    # Load in-cluster Kubernetes configuration but if it fails, load local configuration
    try:
        config.load_incluster_config()
    except config.config_exception.ConfigException:
        config.load_kube_config()

    # Get the current namespace
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        namespace = f.read().strip()

    # Create Kubernetes API client
    api_instance = client.CustomObjectsApi()

    try:
        # Retrieve the route object
        route = api_instance.get_namespaced_custom_object(
            group="route.openshift.io",
            version="v1",
            namespace=namespace,
            plural="routes",
            name=route_name
        )

        # Extract spec.host field
        route_host = route['spec']['host']
        return route_host
    
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    import time

    pipeline_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_package_path
    )

    # Take token and kfp_endpoint as optional command-line arguments
    token = sys.argv[1] if len(sys.argv) > 1 else None
    kfp_endpoint = sys.argv[2] if len(sys.argv) > 2 else None

    if not token:
        print("Token endpoint not provided finding it automatically.")
        token = get_token()

    if not kfp_endpoint:
        print("KFP endpoint not provided finding it automatically.")
        kfp_endpoint = get_route_host(route_name="ds-pipeline-dspa")

    # Pipeline name
    pipeline_name = os.path.basename(__file__).replace('.py', '')

    # If both kfp_endpoint and token are provided, upload the pipeline
    if kfp_endpoint and token:
        client = kfp.Client(host=kfp_endpoint, existing_token=token)

        # If endpoint doesn't have a protocol (http or https), add https
        if not kfp_endpoint.startswith("http"):
            kfp_endpoint = f"https://{kfp_endpoint}"

        try:
            # Get the pipeline by name
            print(f"Pipeline name: {pipeline_name}")
            existing_pipeline = get_pipeline_by_name(client, pipeline_name)
            if existing_pipeline:
                print(f"Pipeline {existing_pipeline.pipeline_id} already exists. Uploading a new version.")
                # Upload a new version of the pipeline with a version name equal to the pipeline package path plus a timestamp
                pipeline_version_name=f"{pipeline_name}-{int(time.time())}"
                client.upload_pipeline_version(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_id=existing_pipeline.pipeline_id,
                    pipeline_version_name=pipeline_version_name
                )
                print(f"Pipeline version uploaded successfully to {kfp_endpoint}")
            else:
                print(f"Pipeline {pipeline_name} does not exist. Uploading a new pipeline.")
                print(f"Pipeline package path: {pipeline_package_path}")
                # Upload the compiled pipeline
                client.upload_pipeline(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_name=pipeline_name
                )
                print(f"Pipeline uploaded successfully to {kfp_endpoint}")
        except Exception as e:
            print(f"Failed to upload the pipeline: {e}")
    else:
        print("KFP endpoint or token not provided. Skipping pipeline upload.")