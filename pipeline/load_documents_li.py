# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# Pipeline to load documents from an S3 bucket into Milvus using `llamaindex` => li

import os
import re
import sys

import kfp

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, OutputPath

from kfp import kubernetes

from kubernetes import client, config

# This component downloads the evaluation data, scaler and model from an S3 bucket and saves it to the correspoding output paths.
# The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
# - SCALER_S3_KEY
# - EVALUATION_DATA_S3_KEY
# - MODEL_S3_KEY
# The data is in pickel format and the file name is passed as an environment variable S3_KEY.
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def get_evaluation_kit(
    evaluation_data_output_dataset: Output[Dataset],
    scaler_output_model: Output[Model],
    model_output_model: Output[Model]
):
    import boto3
    import botocore
    import os
    import zipfile
    import shutil

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')
    evaluation_kit_s3_key = os.environ.get('EVALUATION_KIT_S3_KEY')

    evaluation_data_zip_path = os.environ.get('EVALUATION_DATA_ZIP_PATH')
    scaler_zip_path = os.environ.get('SCALER_ZIP_PATH')
    model_zip_path = os.environ.get('MODEL_ZIP_PATH')

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

    # Create a temporary directory to store the evaluation kit
    
    local_tmp_dir = '/tmp/get_evaluation_kit'
    print(f"local_tmp_dir: {local_tmp_dir}")
    
    # Ensure local_tmp_dir exists
    if not os.path.exists(local_tmp_dir):
        os.makedirs(local_tmp_dir)

    # Get the file name from the S3 key
    file_name = os.path.basename(evaluation_kit_s3_key)    
    # Download the evaluation kit
    local_file_path = f'{local_tmp_dir}/{file_name}'
    print(f"Downloading {evaluation_kit_s3_key} to {local_file_path}")
    bucket.download_file(evaluation_kit_s3_key, local_file_path)
    print(f"Downloaded {evaluation_kit_s3_key}")

    # Unzip the evaluation kit using zipfile module
    extraction_dir = f'{local_tmp_dir}/evaluation_kit'
    print(f"Extracting {local_file_path} in {extraction_dir}")
    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    print(f"Extracted {local_file_path} in {extraction_dir}")

     # Copy the evaluation evaluation_kit/model.onnx to the model output path
    print(f"Copying {extraction_dir}/{model_zip_path} to {model_output_model.path}")
    shutil.copy(f'{extraction_dir}/{model_zip_path}', model_output_model.path)
    
    # Copy the evaluation evaluation_kit/scaler.pkl to the scaler output path
    print(f"Copying {extraction_dir}/{scaler_zip_path} to {scaler_output_model.path}")
    shutil.copy(f'{extraction_dir}/{scaler_zip_path}', scaler_output_model.path)

    # Copy the evaluation evaluation_kit/evaluation_data.pkl to the evaluation data output path
    print(f"Copying {extraction_dir}/{evaluation_data_zip_path} to {evaluation_data_output_dataset.path}")
    shutil.copy(f'{extraction_dir}/{evaluation_data_zip_path}', evaluation_data_output_dataset.path)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["onnx==1.16.1", "onnxruntime==1.18.0", "scikit-learn==1.5.0", "numpy==1.24.3", "pandas==2.2.2"]
)
def test_model(
    evaluation_data_input_dataset: Input[Dataset],
    scaler_input_model: Input[Model],
    model_input_model: Input[Model],
    results_output_metrics: Output[Metrics]
):
    import numpy as np
    import pickle
    import onnxruntime as rt

    # Load the evaluation data and scaler
    with open(evaluation_data_input_dataset.path, 'rb') as handle:
        (X_test, y_test) = pickle.load(handle)
    with open(scaler_input_model.path, 'rb') as handle:
        scaler = pickle.load(handle)

    sess = rt.InferenceSession(model_input_model.path, providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y_pred_temp = sess.run([output_name], {input_name: scaler.transform(X_test.values).astype(np.float32)}) 
    y_pred_temp = np.asarray(np.squeeze(y_pred_temp[0]))
    threshold = 0.995
    y_pred = np.where(y_pred_temp > threshold, 1, 0)

    accuracy = np.sum(np.asarray(y_test) == y_pred) / len(y_pred)
    # print("Accuracy: " + str(accuracy))

    results_output_metrics.log_metric("accuracy", accuracy)

# This component parses the metrics and extracts the accuracy
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301"
)
def parse_metrics(metrics_input: Input[Metrics], accuracy_output: OutputPath(float)):
    print(f"metrics_input: {dir(metrics_input)}")
    accuracy = metrics_input.metadata["accuracy"]
    with open(accuracy_output, 'w') as f:
        f.write(str(accuracy))

# This component uploads the model to an S3 bucket. The connection to the S3 bucket is created using this environment variables:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION
# - AWS_S3_BUCKET
# - AWS_S3_ENDPOINT
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(input_model: Input[Model]):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("MODEL_S3_KEY")

    print(f"Uploading {input_model.path} to {s3_key} in {bucket_name} bucket in {endpoint_url} endpoint")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model.path, s3_key)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["kubernetes"]
)
def refresh_deployment(deployment_name: str):
    import datetime
    import kubernetes

    # Use the in-cluster config
    kubernetes.config.load_incluster_config()

    # Get the current namespace
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
        namespace = f.read().strip()

    # Create Kubernetes API client
    api_instance = kubernetes.client.CustomObjectsApi()

    # Define the deployment patch
    patch = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}"
                    }
                }
            }
        }
    }

    try:
        # Patch the deployment
        api_instance.patch_namespaced_custom_object(
            group="apps",
            version="v1",
            namespace=namespace,
            plural="deployments",
            name=deployment_name,
            body=patch
        )
        print(f"Deployment {deployment_name} patched successfully")
    except Exception as e:
        print(f"Failed to patch deployment {deployment_name}: {e}")

# This pipeline will download evaluation data, download the model, test the model and if it performs well, 
# upload the model to the runtime S3 bucket and refresh the runtime deployment.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(accuracy_threshold: float = 0.95, deployment_name: str = "modelmesh-serving-fraud-detection-model-server",  enable_caching: bool = False):
    # Get the evaluation data, scaler and model
    get_evaluation_kit_task = get_evaluation_kit().set_caching_options(False)

    # Test the model
    test_model_task = test_model(
        evaluation_data_input_dataset=get_evaluation_kit_task.outputs["evaluation_data_output_dataset"],
        scaler_input_model=get_evaluation_kit_task.outputs["scaler_output_model"], 
        model_input_model=get_evaluation_kit_task.outputs["model_output_model"]
    ).set_caching_options(False)

    # Parse the metrics and extract the accuracy
    parse_metrics_task = parse_metrics(metrics_input=test_model_task.outputs["results_output_metrics"]).set_caching_options(False)
    accuracy = parse_metrics_task.outputs["accuracy_output"]

    # Use the parsed accuracy to decide if we should upload the model
    # Doc: https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/execute-kfp-pipelines-locally/
    with dsl.If(accuracy >= accuracy_threshold):
        upload_model_task = upload_model(input_model=get_evaluation_kit_task.outputs["model_output_model"]).after(parse_metrics_task).set_caching_options(False)

        # Setting environment variables for upload_model_task
        upload_model_task.set_env_variable(name="MODEL_S3_KEY", value="models/fraud/1/model.onnx")
        kubernetes.use_secret_as_env(
            task=upload_model_task,
            secret_name='aws-connection-model-runtime',
            secret_key_to_env={
                'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
                'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
                'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
                'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
            }
        )

        # Refresh the deployment
        refresh_deployment(deployment_name=deployment_name).after(upload_model_task).set_caching_options(False)

    # Set the S3 keys for get_evaluation_kit_task and kubernetes secret to be used in the task
    get_evaluation_kit_task.set_env_variable(name="EVALUATION_KIT_S3_KEY", value="models/evaluation_kit.zip")
    get_evaluation_kit_task.set_env_variable(name="EVALUATION_DATA_ZIP_PATH", value="artifact/test_data.pkl")
    get_evaluation_kit_task.set_env_variable(name="SCALER_ZIP_PATH", value="artifact/scaler.pkl")
    get_evaluation_kit_task.set_env_variable(name="MODEL_ZIP_PATH", value="models/fraud/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=get_evaluation_kit_task,
        secret_name='aws-connection-model-staging',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
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