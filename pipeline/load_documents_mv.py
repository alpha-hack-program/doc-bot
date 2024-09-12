# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# Pipeline to load documents from an S3 bucket into Milvus using pymilvus => mv

from math import exp
import os
from pydoc import cli
import re
import sys

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset

from kfp import kubernetes

from kubernetes import client, config
from sympy import N

# # Chunk the text of a document into smaller chunks
# def chunk_text(doc, chunk_size=1024, chunk_overlap=40):
#     doc_chunks = []
#     for i in range(0, len(doc["text"]), chunk_size):
#         start_index = i
#         end_index = i + chunk_size + chunk_overlap if (i + chunk_size + chunk_overlap) < len(doc["text"]) else len(doc["text"]) - 1
#         doc_chunk = {
#             'source': doc["source"],
#             'page_count': doc["page_count"],
#             'text': doc["text"][i:i + chunk_size + chunk_overlap],
#             'dossier': doc["dossier"],
#             'page_start': find_page_num(doc["page_nums"], start_index),
#             'page_end': find_page_num(doc["page_nums"], end_index),
#         }

#         doc_chunks.append(doc_chunk)
#     return doc_chunks

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
    packages_to_install=["boto3", "botocore", "langchain-community", "pypdf"]
)
def get_chunks_from_documents(
    chunk_size: int,
    chunk_overlap: int,
    chunks_output_dataset: Output[Dataset]
):
    import os
    import re
    import pickle

    import boto3
    import botocore

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Find the page number for a given index
    def find_page_num(page_nums, index):
        for i, (start, end) in enumerate(page_nums):
            if index >= start and index <= end:
                return i
        print(f"\n\nERROR: Could not find page number for index {index}.\n\n")
        return 99999
    
    # Chunk the text of a document into smaller chunks using RecursiveCharacterTextSplitter
    def chunk_text_lc(doc, text_splitter: RecursiveCharacterTextSplitter):
        print(f"Chunking {doc['source']}... in chunk_text_lc")
        doc_splits = text_splitter.split_text(doc["text"])
        print(f"doc_splits={len(doc_splits)}")
        start_index = 0
        for i, split in enumerate(doc_splits):
            end_index = len(split) - 1
            doc_splits[i] = {
                'source': doc["source"],
                'page_count': doc["page_count"],
                'text': split,
                'dossier': doc["dossier"],
                'page_start': find_page_num(doc["page_nums"], start_index),
                'page_end': find_page_num(doc["page_nums"], end_index),
            }
            start_index = end_index + 1
        return doc_splits
    
    # Load PDFs from a directory and return a list of dictionaries containing the text and metadata
    def load_pdfs_from_directory(directory_path):
        from pypdf import PdfReader

        docs = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    doc = {
                        'source': filename,
                        'page_count': pdf_reader.get_num_pages(),
                        'text': '',
                        'page_nums': []
                    }
                    index = 0
                    for page_num in range(pdf_reader.get_num_pages()):
                        doc["text"] += pdf_reader.get_page(page_num).extract_text()
                        doc["page_nums"].append((index, len(doc["text"]) - 1))
                        index = len(doc["text"])
                    docs.append(doc)
        return docs

    # Print the chunk_size and chunk_overlap
    print(f"chunk_size: {chunk_size} type: {type(chunk_size)}")
    print(f"chunk_overlap: {chunk_overlap} type: {type(chunk_overlap)}")

    # Get the S3 bucket connection details
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    # Print the connection details
    print(f"endpoint_url: {endpoint_url}")
    print(f"region_name: {region_name}")
    print(f"bucket_name: {bucket_name}")

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

    print(f">>>> bucket: {bucket}")

    # Check if the bucket exists, and create it if it doesn't
    # if not bucket.creation_date:
    #     print(f">>>> Creating bucket {bucket_name}")
    #     s3_resource.create_bucket(Bucket=bucket_name)

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

    # Load PDFs from the directory
    docs = load_pdfs_from_directory(local_tmp_dir)
    print(f"Loaded {len(docs)} PDFs from {local_tmp_dir}.")

    # Print page_count and page_nums for each document
    for doc in docs:
        print(f"source={doc['source']}, page_count={doc['page_count']}, page_nums={doc['page_nums']}, counted={len(doc['page_nums'])}")

    # Add metadata to the documents
    pattern = re.compile(r'^([0-9]+)-(.*)\.pdf$', re.IGNORECASE)
    for doc in docs:
        match = pattern.match(doc["source"])
        if match:
            doc["dossier"] = match.group(1)
        else:
            doc["dossier"] = "Unknown"

    # Prepare the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Chunk the text of each document
    all_doc_chunks = []
    for doc in docs:
        print(f"Chunking {doc['source']}...")
        doc_chunks = chunk_text_lc(doc, text_splitter)
        print(f"Chunked {len(doc_chunks)} chunks from {doc['source']} chunk type =  {type(doc)}.")
        all_doc_chunks.extend(doc_chunks)

    print(f"Total of {len(all_doc_chunks)} chunks of type {type(all_doc_chunks)}.")

    # Save the chunks to the output dataset path
    print(f"Dumping chunks to {chunks_output_dataset.path}")
    with open(chunks_output_dataset.path, "wb") as f:
        pickle.dump(all_doc_chunks, f)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["pymilvus", "transformers", "torch", "langchain_core", "einops"]
)
def add_chunks_to_milvus(
    model_name: str,
    milvus_collection_name: str,
    chunks_input_dataset: Input[Dataset]
):
    import os
    import pickle
    from langchain_core.documents import Document
    from transformers import AutoTokenizer, AutoModel
    import torch
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

    # Generate an embedding for a given text
    def generate_embedding(text: str, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

    # Collection name
    print(f"milvus_collection_name: {milvus_collection_name}")

    # Get the Mivus connection details
    milvus_host = os.environ.get('MILVUS_HOST')
    milvus_port = os.environ.get('MILVUS_PORT')
    milvus_username = os.environ.get('MILVUS_USERNAME')
    milvus_password = os.environ.get('MILVUS_PASSWORD')
    
    # Print the connection details
    print(f"milvus_host: {milvus_host}")
    print(f"milvus_port: {milvus_port}")

    # Initialize Hugging Face model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model_kwargs = {'trust_remote_code': True}
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True) # TODO: Add trust_remote_code=True to avoid warnings
    vector_size = model.config.hidden_size

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
    
    if not isinstance(chunks, list): #  or not all(isinstance(doc, Document) for doc in chunks):
        raise TypeError("The loaded data is not a List[langchain_core.documents.Document].")

    print(f"Loaded {len(chunks)} chunks")

    # Connect to Milvus
    connections.connect("default", host=milvus_host, port=milvus_port, user=milvus_username, password=milvus_password)

    # Drop collections before creating them?
    drop_before_create = True

    # Delete the chunks collections if it already exists and drop_before_create is True
    if utility.has_collection(milvus_collection_name) and drop_before_create:
        utility.drop_collection(milvus_collection_name)

    # Define the chunks collection schema
    milvus_collection_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="page_start", dtype=DataType.INT16),
        FieldSchema(name="page_end", dtype=DataType.INT16),
        FieldSchema(name="dossier", dtype=DataType.VARCHAR, max_length=128),
    ]
    chunks_collection_schema = CollectionSchema(milvus_collection_fields, "Schema for storing document chunks.")

    # Create a collection for chunks
    milvus_chunks_collection = None
    if utility.has_collection(milvus_collection_name):
        milvus_chunks_collection = Collection(milvus_collection_name)
    else:
        milvus_chunks_collection = Collection(milvus_collection_name, chunks_collection_schema)

    # Prepare data for insertion
    vectors = [generate_embedding(doc_chunk["text"], tokenizer, model) for doc_chunk in chunks]

    # Extract attr from the doc_chunks and store them in separate lists
    sources = [doc_chunk['source'] for doc_chunk in chunks]
    texts = [doc_chunk['text'] for doc_chunk in chunks]
    page_starts = [doc_chunk['page_start'] for doc_chunk in chunks]
    page_ends = [doc_chunk['page_end'] for doc_chunk in chunks]
    dossiers = [doc_chunk['dossier'] for doc_chunk in chunks]

    # Insert data into the collection
    entities = [
        vectors,
        sources,
        texts,
        page_starts,
        page_ends,
        dossiers,
    ]

    # Print shape of the entities
    print(f"Entities shape: {len(entities)} x {len(entities[0])} : {len(entities[1])} : {len(entities[2])} : {len(entities[3])} : {len(entities[4])} : {len(entities[5])}")

    insert_result = milvus_chunks_collection.insert(entities)
    milvus_chunks_collection.flush()

    print(f"Inserted {len(insert_result.primary_keys)} records into Milvus.")

    # Add an index to the collection
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"efConstruction": 200, "M": 16}  # Typical values for efConstruction and M
    }
    milvus_chunks_collection.create_index(field_name="vector", index_params=index_params)

    # Load the collection
    milvus_chunks_collection.load()


# This pipeline will download evaluation data, download the model, test the model and if it performs well, 
# upload the model to the runtime S3 bucket and refresh the runtime deployment.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(model_name: str = "nomic-ai/nomic-embed-text-v1", 
             collection_name: str = "chunks", 
             chunk_size: int = 2048, 
             chunk_overlap: int = 200,  
             enable_caching: bool = False):
    
    # Get all the PDFs from the S3 bucket
    get_chunks_from_documents_task = get_chunks_from_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap).set_caching_options(False)

    # Add chunks to vector store
    add_chunks_to_vector_store_task = add_chunks_to_milvus(
        model_name=model_name,
        milvus_collection_name=collection_name,
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
        # If endpoint doesn't have a protocol (http or https), add https
        if not kfp_endpoint.startswith("http"):
            kfp_endpoint = f"https://{kfp_endpoint}"

        # Create a Kubeflow Pipelines client
        client = kfp.Client(host=kfp_endpoint, existing_token=token)

        try:
            result = None
            # Get the pipeline by name
            print(f"Pipeline name: {pipeline_name}")
            existing_pipeline = get_pipeline_by_name(client, pipeline_name)
            if existing_pipeline:
                print(f"Pipeline {existing_pipeline.pipeline_id} already exists. Uploading a new version.")
                # Upload a new version of the pipeline with a version name equal to the pipeline package path plus a timestamp
                pipeline_version_name=f"{pipeline_name}-{int(time.time())}"
                result = client.upload_pipeline_version(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_id=existing_pipeline.pipeline_id,
                    pipeline_version_name=pipeline_version_name
                )
                print(f"Pipeline version uploaded successfully to {kfp_endpoint}")
            else:
                print(f"Pipeline {pipeline_name} does not exist. Uploading a new pipeline.")
                print(f"Pipeline package path: {pipeline_package_path}")
                # Upload the compiled pipeline
                result = client.upload_pipeline(
                    pipeline_package_path=pipeline_package_path,
                    pipeline_name=pipeline_name
                )
                print(f"Pipeline uploaded successfully to {kfp_endpoint}")

            # Recurring execution of the pipeline
            try:
                from datetime import datetime, timezone

                # Get the default experiment ID
                default_experiment = client.get_experiment(experiment_name='Default')

                # Check if the recurring run already exists
                job_name = f'job-{pipeline_name}'
                recurring_runs_response = client.list_recurring_runs(experiment_id=default_experiment.experiment_id)
                # print(f"Recurring runs: {recurring_runs_response.recurring_runs}")
                current_recurring_run = None
                if recurring_runs_response.recurring_runs:
                    for run in recurring_runs_response.recurring_runs:
                        # print(f"Run: {run}")
                        # print(f">>>> Run: {run.display_name}")
                        if run.display_name == job_name:
                            print(f"Recurring run {run.display_name} already exists for {pipeline_name}.")
                            current_recurring_run = run
                            break

                # If the recurring run exists, delete it
                if current_recurring_run:
                    print(f"Deleting the recurring run {current_recurring_run.recurring_run_id} for {pipeline_name}.")
                    client.delete_recurring_run(current_recurring_run.recurring_run_id)

                # Specify parameters for the recurring run
                experiment_id = default_experiment.experiment_id
                # job_name = f'job-{pipeline_name}'
                start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                print(f"Start time: {start_time}")
                interval_second = 300  # Run every hour
                if result and result.pipeline_id and result.pipeline_version_id:
                    print(f"Creating a pipeline run for pipeline {result.pipeline_id}")
                    client.create_recurring_run(
                        experiment_id=experiment_id,
                        job_name=job_name,
                        pipeline_id=result.pipeline_id,
                        version_id=result.pipeline_version_id,
                        start_time=start_time,
                        interval_second=interval_second
                    )
                
                print(f"Recurring Pipeline run created successfully for {pipeline_name}.")
            except Exception as e:
                print(f"Failed to create the pipeline run: {e}")
        except Exception as e:
            print(f"Failed to upload the pipeline: {e}")

        
    else:
        print("KFP endpoint or token not provided. Skipping pipeline upload.")