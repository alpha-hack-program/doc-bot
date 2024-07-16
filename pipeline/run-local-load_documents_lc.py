# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import local
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from load_documents_lc import pipeline, get_chunks_from_documents

local.init(runner=local.SubprocessRunner())

os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
os.environ['AWS_DEFAULT_REGION'] = 'none'
os.environ['AWS_S3_ENDPOINT'] = 'https://minio-s3-ic-shared-minio.apps.cluster-msq7f.sandbox3206.opentlc.com'
os.environ['AWS_S3_BUCKET'] = 'documents'

os.environ['MILVUS_HOST'] = 'localhost'
os.environ['MILVUS_PORT'] = '19530'
os.environ['MILVUS_COLLECTION'] = 'documents'
os.environ['MILVUS_USERNAME'] = 'root'
os.environ['MILVUS_PASSWORD'] = 'Milvus'


# evaluation_data_output_dataset = Dataset( name='evaluation_data_output_dataset',
#                                              uri='/Users/cvicensa/Projects/openshift/alpha-hack-program/ai-studio-rhoai/pipeline/local_outputs/deploy-2024-06-26-11-18-35-229772/get-evaluation-kit/evaluation_data_output_dataset',
#                                              metadata={} )
# output_model= Model(name='output_model',
#                     uri='/Users/cvicensa/Projects/openshift/alpha-hack-program/ai-studio-rhoai/pipeline/local_outputs/deploy-2024-06-26-11-18-35-229772/get-evaluation-kit/output_model',
#                     metadata={} )
# scaler_output_model = Model( name='scaler_output_model',
#                                 uri='/Users/cvicensa/Projects/openshift/alpha-hack-program/ai-studio-rhoai/pipeline/local_outputs/deploy-2024-06-26-11-18-35-229772/get-evaluation-kit/scaler_output_model',
#                                 metadata={} )

# get_chunks_from_documents_task = get_chunks_from_documents()

# run pipeline
pipeline_task = pipeline(model_name="sentence-transformers/all-MiniLM-L6-v2")

