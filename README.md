# Doc Bot Hacking Sprint

<table>
    <tr>
        <td><b>Title:</b></td>
        <td>Doc Bot</td>
    </tr>
    <tr>
        <td><b>Goal:</b></td>
        <td>Deploy RHOAI LLM <a href="https://ai-on-openshift.io/demos/llm-chat-doc/llm-chat-doc/#rag-chatbot-full-walkthrough">tutorial</a> and learn the inner workings of the system.</td>
    </tr>
    <tr>
        <td><b>Output:</b></td>
        <td>Being able to feed and deploy a RAG based chatbot or at least the backend/API of it.</td>
    </tr>
    <tr>
        <td><b>Timing:</b></td>
        <td>2 to 3h</td>
    </tr>
    <tr>
        <td><b>Notes:</b></td>
        <td>We'll start with the RHOAI Insurance Claim RHPDS demo.</td>
    </tr>
</table>

# Starting point of this hacking sprint

[Chat with your documentation](https://ai-on-openshift.io/demos/llm-chat-doc/llm-chat-doc/) lab from [AI on OpenShift](https://ai-on-openshift.io/).

# Steps

## Model Serving

Deploy vLLM Model Serving instance in the OpenAI compatible API mode, either:

- [as a custom server runtime in ODH/RHOAI](https://github.com/atarazana/doc-bot/blob/main/serving-runtimes/vllm_runtime/README.md)
- [as a standalone server in OpenShift](https://github.com/atarazana/doc-bot/blob/main/llm-servers/vllm/README.md)

### As a runtime

You must first make sure that you have properly installed the necessary component of the Single-Model Serving stack, as documented [here](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2-latest/html/serving_models/serving-large-models_serving-large-models).

From the documentation:

> For deploying large models such as large language models (LLMs), OpenShift AI includes a single-model serving platform that is based on the KServe component. Because each model is deployed on its own model server, the single-model serving platform helps you to deploy, monitor, scale, and maintain large models that require increased resources.
> - **KServe:** A Kubernetes custom resource definition (CRD) that orchestrates model serving for all types of models. KServe includes model-serving runtimes that implement the loading of given types of model servers. KServe also handles the lifecycle of the deployment object, storage access, and networking setup.
> - **Red Hat OpenShift Serverless:** A cloud-native development model that allows for serverless deployments of models. OpenShift Serverless is based on the open source Knative project.
> - **Red Hat OpenShift Service Mesh:** A service mesh networking layer that manages traffic flows and enforces access policies. OpenShift Service Mesh is based on the open source Istio project.

### As a deployment [optional]

TODO

## Preparation

> If you need to delete RHOAI go [here](./docs/Uninstall.md)

Install the following operators prior to install OpenShift AI operator:

- OpenShift Serverless
- OpenShift Service Mesh

Now, in order to deploy `KServe` using the same certificate OpenShift cluster is using for routers, do the following:

> Intructions from: https://ai-on-openshift.io/odh-rhoai/single-stack-serving-certificate/

Get the name of the secret used for routing tasks in your OpenShift cluster:

```sh
export INGRESS_SECRET_NAME=$(oc get ingresscontroller default -n openshift-ingress-operator -o json | jq -r .spec.defaultCertificate.name)

oc get secret ${INGRESS_SECRET_NAME} -n openshift-ingress -o yaml > rhods-internal-primary-cert-bundle-secret.yaml
```

Clean `rhods-internal-primary-cert-bundle-secret.yaml`, change name to `rhods-internal-primary-cert-bundle-secret` and type to `kubernetes.io/tls`, it should look like:

```yaml
kind: Secret
apiVersion: v1
metadata:
name: rhods-internal-primary-cert-bundle-secret
data:
tls.crt: >-
    LS0tLS1CRUd...
tls.key: >-
    LS0tLS1CRUd...
type: kubernetes.io/tls
```

Create the secret in `istio-system`:

```sh
oc apply -n istio-system -f rhods-internal-primary-cert-bundle-secret.yaml
```

Check the secret is in place:

```sh
oc get secret rhods-internal-primary-cert-bundle-secret -n istio-system
```

Install OpenShift AI operator.

```sh
cat << EOF| oc create -f -
---
apiVersion: v1
kind: Namespace
metadata:
  name: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator 
spec:
  name: rhods-operator
  channel: stable-2.10
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Manual
EOF
```

There should be only one install plan!

```sh
oc get installplans -n redhat-ods-operator -o name
```

Patch it in order to approve it:

```sh
for installplan in $(oc get installplans -n redhat-ods-operator -o name); do
  oc patch $installplan -n redhat-ods-operator --type merge --patch '{"spec":{"approved":true}}'
done
```

Finally create the `DSC` object (RHOAI CRD) which should look like [this](./examples/cluster/default-dsc.yaml):

```sh
cat << EOF| oc create -f -
---
kind: DataScienceCluster
apiVersion: datasciencecluster.opendatahub.io/v1
metadata:
  name: default-dsc
  labels:
    app.kubernetes.io/name: datasciencecluster
    app.kubernetes.io/instance: default-dsc
    app.kubernetes.io/part-of: rhods-operator
    app.kubernetes.io/managed-by: kustomize
    app.kubernetes.io/created-by: rhods-operator
spec:
  components:
    codeflare:
      managementState: Managed
    dashboard:
      managementState: Managed
    datasciencepipelines:
      managementState: Managed
    kserve:
      managementState: Managed
      serving:
        ingressGateway:
          certificate:
            # Intructions from: https://ai-on-openshift.io/odh-rhoai/single-stack-serving-certificate/
            # You have to copy the secret create it in istio-system namespace
            secretName: rhods-internal-primary-cert-bundle-secret
            type: Provided
        managementState: Managed
        name: knative-serving
    modelmeshserving:
      managementState: Managed
    kueue:
      managementState: Managed
    ray:
      managementState: Managed
    workbenches:
      managementState: Managed
EOF
```

## Deploying the vLLM runtime

Once the stack is installed, adding the runtime is pretty straightforward:

- As an admin, in the OpenShift AI Dashboard, open the menu `Settings -> Serving runtimes`.
- Click on `Add serving runtime`.
- For the type of model serving platforms this runtime supports, select `Single model serving platform`.
- Upload the file `./vllm_runtime/vllm-runtime.yaml`, or click `Start from scratch` and copy/paste its content.

The runtime is now available when deploying a model.

## Deploying the model

In both cases, deploy the model `mistralai/Mistral-7B-Instruct-v0.2`. The model can be found [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

> **NOTE:** https://huggingface.co/docs/hub/models-downloading

Some hints:
> [Click here to install `git lfs`](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 

Get a PAT from HF and put it in `.hf-token`.

```sh
export HF_USERNAME=<YOU_USER_NAME>
export HF_TOKEN=$(cat .hf-token)
export MODEL_ROOT="mistralai"
export MODEL_ID="Mistral-7B-Instruct-v0.2"
export HF_TMPDIR="tmp"

mkdir -p ${HF_TMPDIR} && cd ${HF_TMPDIR}
git lfs install
git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${MODEL_ROOT}/${MODEL_ID}
rm -rf ${MODEL_ID}/.git
```

Flan T5

```sh
export HF_USERNAME=<YOU_USER_NAME>
export HF_TOKEN=$(cat .hf-token)
export MODEL_ROOT="google"
export MODEL_ID="flan-t5-small"
export HF_TMPDIR="tmp"

mkdir -p ${HF_TMPDIR} && cd ${HF_TMPDIR}
git lfs install
git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${MODEL_ROOT}/${MODEL_ID}
rm -rf ${MODEL_ID}/.git
```

### Copy the model files to an S3 bucket

Upload to an S3 bucket. Run this commands from `${HF_TMPDIR}`:

```sh
export AWS_S3_BUCKET=models
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_DEFAULT_REGION=none  # Any value is fine
export AWS_S3_ENDPOINT=localhost:9000  # e.g., http://localhost:9000
export AWS_S3_CUSTOM_DOMAIN=${AWS_S3_ENDPOINT}
export AWS_S3_USE_PATH_STYLE=1

# Create a bucket
aws s3api create-bucket --bucket ${AWS_S3_BUCKET} --endpoint-url "https://${AWS_S3_ENDPOINT}" 

# Upload the model to files folder in the bucket
aws s3 sync ${MODEL_ID} s3://${AWS_S3_BUCKET}/${MODEL_ROOT}/${MODEL_ID}/ --endpoint-url "https://${AWS_S3_ENDPOINT}" 
```

### Deploy the mistral to vLLM

This runtime can be used in the exact same way as the out of the box ones. This is the sequence of actions to use our custom `vLLM` runtime:

> **Remember** that the custom runtime has to be installed before proceeding with this instructions.

1. Create a project in RHOAI
2. Create a connection to the S3 bucket that contains the model files.
3. Deploy the model from the Dashboard.
   > Make sure you have added a GPU to your GPU configuration, that you have enough VRAM (GPU memory) to load the model, and that you have enough standard memory (RAM). Although the model loads into the GPU, RAM is still used for the pre-loading operations.
4. Once the model is loaded, you can access the inference endpoint provided through the dashboard.

#### Create a project in RHOAI

Click on `Create data science project`.

![Create DS Project](./images/create-ds-project-1.png)

Give it a name.

![Create DS Project](./images/create-ds-project-2.png)

Or run this:

```sh
PROJECT_NAME="vllm-mistral-7b"
DISPLAY_NAME="vLLM Mistral 7B"
DESCRIPTION="Mistral Model run using vLLM"

oc new-project $PROJECT_NAME --display-name="$DISPLAY_NAME" --description="$DESCRIPTION"

oc label namespace $PROJECT_NAME opendatahub.io/dashboard=true
```

#### Create a data connection

Click on `Add data connection`.

![Create Data Connection](./images/add-data-connection.png)

**Name for your connection:** mistral

Or run this:

```sh
CONNECTION_NAME=mistral
AWS_S3_ENDPOINT=http://minio.ic-shared-minio.svc:9000

oc create secret generic -n ${PROJECT_NAME} aws-connection-${CONNECTION_NAME} \
--from-literal=AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
--from-literal=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
--from-literal=AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
--from-literal=AWS_S3_ENDPOINT=${AWS_S3_ENDPOINT} \
--from-literal=AWS_S3_BUCKET=${AWS_S3_BUCKET}

oc label secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/dashboard=true
oc label secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/managed=true
oc annotate secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/connection-type=s3
oc annotate secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME openshift.io/display-name=${CONNECTION_NAME}
```

#### Deploy the model from the Dashboard

Click on `Deploy Model` in Single-model serving platform.

![Deploy LLM](./images/deploy-model-on-vllm-1.png)

Use the following data to deploy your model:

- **Model name:** mistral-7b
- **Serving runtime:** mistral-7b
- **Model framework (name - version):** pytorch
- **Model server size:** Custom
  - *CPUs requested:* 6 cores
  - *Memory requested:* 24 Gi
  - *CPUs limit:* 8 cores
  - *Memory limit:* 24 Gi
- **Accelerator:** NVIDIA GPU
- **Number of accelerators:** 1
- **Existing data connection:**
  - *Name:* mistral
  - *Path:* mistralai/Mistral-7B-Instruct-v0.2

Path should be `${MODEL_ROOT}/${MODEL_ID}`.

Or you can run this command:

```sh
oc create -n ${PROJECT_NAME} -f ./vllm_runtime/vllm-mistral-7b-instance.yaml
```

> It's worth mentioning that we have added a node selector to move this workload to a node with A10G gpu.
> ```yaml
>apiVersion: serving.kserve.io/v1beta1
>kind: InferenceService
>metadata:
>  annotations:
>    openshift.io/display-name: mistral-7b
>    serving.knative.openshift.io/enablePassthrough: 'true'
>    sidecar.istio.io/inject: 'true'
>    sidecar.istio.io/rewriteAppHTTPProbers: 'true'
>  name: mistral-7b
>  labels:
>    opendatahub.io/dashboard: 'true'
>spec:
>  predictor:
>    maxReplicas: 1
>    minReplicas: 1
>    model:
>      modelFormat:
>        name: pytorch
>      name: ''
>      resources:
>        limits:
>          nvidia.com/gpu: '1'
>        requests:
>          nvidia.com/gpu: '1'
>      runtime: mistral-7b
>      storage:
>        key: aws-connection-mistral
>        path: mistralai/Mistral-7B-Instruct-v0.2
>    tolerations:
>      - effect: NoSchedule
>        key: nvidia.com/gpu
>        operator: Exists
>    nodeSelector:
>      nvidia.com/gpu.product: NVIDIA-A10G # HERE
>```


Wait until the model has been correctly deployed.

> **S3:** `oc logs deploy/mistral-7b-predictor-00001-deployment -c storage-initializer -n ${PROJECT_NAME}`

Check if the predictor url is ready:

```sh
oc get inferenceservice/mistral-7b -n $PROJECT_NAME -o json | jq -r .status.url
```

#### Usage

This implementation of the runtime provides an **OpenAI compatible API**. So any tool or library that can connect to OpenAI services will be able to consume the endpoint.

Python and Curl examples are provided [here](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-completions-api-with-vllm).

You can also find a notebook example using Langchain to query vLLM in this repo [here](../examples/notebooks/langchain/Langchain-vLLM-Prompt-memory.ipynb).

Also, vLLM provides a full Swagger UI where you can get the full documentation of the API (methods, parameters), and try it directly without any coding,... It is accessible at the address `https://your-endpoint-address/docs`.

Example tested on vLLM on RHOAI 2.8:

Get model ID:

```sh
MODEL_NAME="mistral-7b"
VLLM_PREDICTOR_URL=$(oc get inferenceservice/${MODEL_NAME} -n $PROJECT_NAME -o json | jq -r .status.url)

RUNTIME_MODEL_ID=$(curl -ks -X 'GET' "${VLLM_PREDICTOR_URL}/v1/models" -H 'accept: application/json' | jq -r .data[0].id )
echo ${RUNTIME_MODEL_ID}
```

Prompt:

```sh
curl -s -X 'POST' \
  "${VLLM_PREDICTOR_URL}/v1/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "'${RUNTIME_MODEL_ID}'",
  "prompt": "San Francisco is a",
  "max_tokens": 25,
  "temperature": 0
}'
```

### Deploy the mistral to vLLM but using 2x T4 GPUs

#### First let's create a machineset for our purposes

Run `./bootstrap/gpu-machineset.sh` and choose `(2) Tesla T4 Multi GPU` and make sure you get this after typing `2`.

> You have to be `cluster-admin` for this to succeed.

```
### Selected GPU instance type: Tesla T4 Multi GPU
### Selected GPU instance type: g4dn.12xlarge
```

Now choose the AWS region and AZ you want the machine set to deploy workers with GPUs:

> If starting of the RHPDS Claim Service demo you should probably use `us-east-2` and `az1`.

```
### Enter the AWS region (default: us-west-2): us-east-2
### Select the availability zone (az1, az2, az3):
1) az1
2) az2
3) az3
Please enter your choice: 1
### Creating new machineset worker-gpu-g4dn.12xlarge-us-east-2a.
machineset.machine.openshift.io/worker-gpu-g4dn.12xlarge-us-east-2a created
--- New machineset worker-gpu-g4dn.12xlarge-us-east-2a created.
```

Run this command to check if the machineset is in place.

```sh
MACHINESET_NAME=$(oc get machineset -n openshift-machine-api -o json | jq -r '.items[] | select(.metadata.name | test("^worker-gpu")) | .metadata.name')
echo ${MACHINESET_NAME}
```

It should return something like: "worker-gpu-g4dn.12xlarge-us-east-2a"

Check if our new nodes are resdy:

```sh
oc get machineset/${MACHINESET_NAME} -n openshift-machine-api
```

You should get something like this:

```
NAME                                  DESIRED   CURRENT   READY   AVAILABLE   AGE
worker-gpu-g4dn.12xlarge-us-east-2a   1         1         1       1           8m31s
```

#### Create a project in RHOAI

```sh
PROJECT_NAME="vllm-mistral-2x-t4"
DISPLAY_NAME="vLLM Mistral 2x T4"
DESCRIPTION="Mistral Model run using vLLM with 2x T4"

oc new-project $PROJECT_NAME --display-name="$DISPLAY_NAME" --description="$DESCRIPTION"

oc label namespace $PROJECT_NAME opendatahub.io/dashboard=true
```

#### Create a data connection

Click on `Add data connection`.

![Create Data Connection](./images/add-data-connection.png)

**Name for your connection:** mistral

Or run this:

```sh
CONNECTION_NAME=mistral
AWS_S3_ENDPOINT=http://minio.ic-shared-minio.svc:9000

oc create secret generic -n ${PROJECT_NAME} aws-connection-${CONNECTION_NAME} \
--from-literal=AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
--from-literal=AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
--from-literal=AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
--from-literal=AWS_S3_ENDPOINT=${AWS_S3_ENDPOINT} \
--from-literal=AWS_S3_BUCKET=${AWS_S3_BUCKET}

oc label secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/dashboard=true
oc label secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/managed=true
oc annotate secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME opendatahub.io/connection-type=s3
oc annotate secret aws-connection-${CONNECTION_NAME} -n $PROJECT_NAME openshift.io/display-name=${CONNECTION_NAME}
```

#### Deploy the model from the cli

```sh
oc create -n ${PROJECT_NAME} -f ./vllm_runtime/vllm-mistral-7b-multigpu-instance.yaml
```

Wait until the model has been correctly deployed.

> **S3:** `oc logs deploy/mistral-7b-2x-t4-predictor-00001-deployment -c storage-initializer -n ${PROJECT_NAME}`

Check if the predictor url is ready:

```sh
oc get inferenceservice/mistral-7b-2x-t4 -n $PROJECT_NAME -o json | jq -r .status.url
```

Run this command to check if the pod has two T4s:

```sh
T4_POD=$(oc get pod -n $PROJECT_NAME -o json | jq -r .items[0].metadata.name)
oc rsh ${T4_POD} nvidia-smi
```

You should get something like this:

```
Thu Jun  6 17:37:44 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1C.0 Off |                    0 |
| N/A   37C    P0              26W /  70W |  11145MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       On  | 00000000:00:1D.0 Off |                    0 |
| N/A   37C    P0              26W /  70W |  11145MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```

#### Usage

This implementation of the runtime provides an **OpenAI compatible API**. So any tool or library that can connect to OpenAI services will be able to consume the endpoint.

Python and Curl examples are provided [here](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-completions-api-with-vllm).

You can also find a notebook example using Langchain to query vLLM in this repo [here](../examples/notebooks/langchain/Langchain-vLLM-Prompt-memory.ipynb).

Also, vLLM provides a full Swagger UI where you can get the full documentation of the API (methods, parameters), and try it directly without any coding,... It is accessible at the address `https://your-endpoint-address/docs`.

Example tested on vLLM on RHOAI 2.8:

Get model ID:

```sh
MODEL_NAME="mistral-7b-2x-t4"
VLLM_PREDICTOR_URL=$(oc get inferenceservice/${MODEL_NAME} -n $PROJECT_NAME -o json | jq -r .status.url)

RUNTIME_MODEL_ID=$(curl -ks -X 'GET' "${VLLM_PREDICTOR_URL}/v1/models" -H 'accept: application/json' | jq -r .data[0].id )
echo ${RUNTIME_MODEL_ID}
```

Prompt:

```sh
curl -s -X 'POST' \
  "${VLLM_PREDICTOR_URL}/v1/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "'${RUNTIME_MODEL_ID}'",
  "prompt": "San Francisco is a",
  "max_tokens": 25,
  "temperature": 0
}'
```

# Vector Store

## Milvus deployment

For our RAG we will need a Vector Database to store the Embeddings of the different documents. In this example we are using Milvus.

Deployment instructions specific to OpenShift are available [here](./vector-databases/milvus).

After you follow those instructions you should have a Milvus instance ready to be populated with documents.

### Architecture

![Architecture](./images/milvus.png)

### Requirements

- Access to the OpenShift cluster.
- A default StorageClass must be configured.

### Deployment Options

We'll deploy a default installation of Milvus, either standalone or in cluster mode, with authentication enabled. The provided `openshift-values.yaml` file can be changed to configure the installation.

- The default Milvus deployment leverages Minio to store logs and index files. This can be replaced by another S3 storage system
- Default configuration uses Pulsar for managing logs of recent changes, outputting stream logs, and providing log subscriptions. This can be replaced by Kafka

To modify those components, as well many other configuration parameters, please refer to the [configuration documentation](https://milvus.io/docs/deploy_s3.md) and modify the values file according to your needs.

### Deployment procedure

Milvus can be deployed in Standalone or Cluster mode. Cluster mode, leveraging Pulsar, etcd and Minio for data persistency, will bring redundancy, as well as easy scale up and down of the different components.

Although Milvus features an operator to easily deploy it in a Kubernetes environment, this method has not been tested yet, while waiting for the different corrections to be made to the deployment code for OpenShift specificities.

Instead, this deployment method is based on the [Offline installation](https://milvus.io/docs/install_offline-helm.md) that purely rely on Helm Charts.

- Log into your OpenShift cluster, and create a new project to host your Milvus installation:

```bash
MILVUS_PROJECT_NAME=milvus
oc new-project ${MILVUS_PROJECT_NAME}
```

- Add and update Milvus Helm repository locally:

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

- Fetch the file [`openshift-values.yaml`](openshift-values.yaml) from this repo. This file is really important as it sets specific values for OpenShift compatibility. You can also modify some of the values in this file to adapt the deployment to your requirements, notably modify the Minio admin user and password.

    ```bash
    wget https://raw.githubusercontent.com/alpha-hack-program/doc-bot/main/vector-databases/milvus/openshift-values.yaml
    ```

- Create the manifest:
  - For Milvus standalone:

    ```bash
    helm template -f openshift-values.yaml vectordb --set cluster.enabled=false --set etcd.replicaCount=1 --set minio.mode=standalone --set pulsar.enabled=false milvus/milvus > milvus_manifest_standalone.yaml
    ```

  - For Milvus cluster:

    ```bash
    helm template -f openshift-values.yaml vectordb milvus/milvus > milvus_manifest_cluster.yaml
    ```

- **VERY IMPORTANT**: you must patch the generated manifest, as some settings are incompatible with OpenShift. Those commands are using the **[yq tool](https://mikefarah.gitbook.io/yq/)** (beware, the real one, not the Python version):
  - For Milvus Standalone:
  
    ```bash
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.securityContext) = {}' -i milvus_manifest_standalone.yaml
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.containers[0].securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_standalone.yaml
    yq '(select(.kind == "Deployment" and .metadata.name == "vectordb-minio") | .spec.template.spec.securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_standalone.yaml
    ```

  - For Milvus Cluster:
  
    ```bash
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.securityContext) = {}' -i milvus_manifest_cluster.yaml
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.containers[0].securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_cluster.yaml
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-minio") | .spec.template.spec.securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_cluster.yaml
    ```

- Deploy Milvus (eventually change the name of the manifest)!

  - For Milvus Standalone:
  
    ```bash
    oc apply -n ${MILVUS_PROJECT_NAME} -f milvus_manifest_standalone.yaml
    ```

  - For Milvus Cluster:
  
    ```bash
    oc apply -n ${MILVUS_PROJECT_NAME} -f milvus_manifest_cluster.yaml
    ```

- To deploy the management UI for Milvus, called Attu, apply the file [attu-deployment.yaml](attu-deployment.yaml):

```bash
oc apply -f https://raw.githubusercontent.com/alpha-hack-program/doc-bot/main/vector-databases/milvus/attu-deployment.yaml
```

NOTE: Attu deployment could have been done through the Helm chart, but this would not properly create the access Route.

Milvus is now deployed, with authentication enabled. The default and only admin user is `root`, with the default password `Milvus`. Please see the following section to modify this root access and create the necessary users and roles.

## Day-2 operations

Milvus implements a full RBAC system to control access to its databases and collections. It is recommended to:

- Change the default password
- Create collections
- Create users and roles to give read/write or read access to the different collections

All of this can be done either using the PyMilvus library, or the Attu UI deployed in the last step.

Milvus features an [awesome documentation](https://milvus.io/docs), detailing all the configuration, maintenance, and operations procedure. This is a must-read to grasp all aspects of its architecture and usage.

## Usage

Several example notebooks are available to show how to use Milvus:

- Collection creation and document ingestion using Langchain: [Langchain-Milvus-Ingest.ipynb](../../examples/notebooks/langchain/Langchain-Milvus-Ingest.ipynb)
- Collection creation and document ingestion using Langchain with Nomic AI Embeddings: [Langchain-Milvus-Ingest-nomic.ipynb](../../examples/notebooks/langchain/Langchain-Milvus-Ingest-nomic.ipynb)
- Query a collection using Langchain: [Langchain-Milvus-Query.ipynb](../../examples/notebooks/langchain/Langchain-Milvus-Query.ipynb)
- Query a collection using Langchain with Nomic AI Embeddings: [Langchain-Milvus-Query-nomic.ipynb](../../examples/notebooks/langchain/Langchain-Milvus-Query-nomic.ipynb)

# GitOps Deployment of Doc Bot

> **NOTE:** Install OpenShift GitOps prior to run the following commands.

> **TODO! NOTE:** This procedure deploys Milvus and all the other assets needed but the bucket models needs to be in place in the minio instance in `ic-shared-minio` namespace... 





# QA

- How do you know how many GPU is consumed?

# Documentation

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/23.9.2/index.html)
- [How to install vLLM as a RHOAI runtime](https://github.com/atarazana/doc-bot/blob/main/serving-runtimes/vllm_runtime/README.md)
- [How to install vLLM as deployment](https://github.com/atarazana/doc-bot/blob/main/llm-servers/vllm/README.md)
- https://www.3blue1brown.com/lessons/attention
- https://huggingface.co/learn/cookbook/en/advanced_rag
- https://github.com/alvarolop/ocp-ai/blob/main/openshift/rhoai-configuration/OdhDashboardConfig.yaml#L32-L34
- https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html-single/serving_models/index#about-model-serving_about-model-serving
- https://aws.amazon.com/ec2/instance-types/
- https://aws.amazon.com/ec2/pricing/on-demand/
- https://github.com/alvarolop/rhoai-gitops/tree/main

RAG:
- https://huggingface.co/learn/cookbook/advanced_rag
- https://huggingface.co/blog/ray-rag
- https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/

Infra:
- https://github.com/alvarolop/rhoai-gitops.git

RAG Evaluation:
- https://docs.ragas.io/en/latest/index.html
- https://mlflow.org/docs/latest/llms/llm-evaluate/notebooks/rag-evaluation.html
- https://www.trulens.org/
- https://docs.ragas.io/en/latest/getstarted/evaluation.html#get-started-evaluation
- https://github.com/rcarrat-AI/unir_master_thesis/blob/main/notebooks/TFM_Experiments_LLM_vs_RAG.ipynb

GPUs:
- https://github.com/rh-aiservices-bu/gpu-partitioning-guide
- https://github.com/rh-aiservices-bu/multi-gpu-llms

GitOps Iniciatives:
- https://gitlab.consulting.redhat.com/denizbank-ai/mlops-cluster-config
- https://github.com/rh-aiservices-bu/parasol-insurance
- https://github.com/redhat-na-ssa/demo-ai-gitops-catalog

KServe certs:
- https://ai-on-openshift.io/odh-rhoai/single-stack-serving-certificate/#procedure

Embeddings:
- https://old.tacosdedatos.com/word-to-vec-ilustrado

Dataset Embeddings:
- https://huggingface.co/datasets/embedding-data/simple-wiki

Uninstall RHOAI:
- https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html/installing_and_uninstalling_openshift_ai_self-managed/uninstalling-openshift-ai-self-managed_uninstalling-openshift-ai-self-managed

Milvus:
- https://milvus.io/tools/sizing

# Pipelines KFP

https://www.kubeflow.org/docs/components/pipelines/v2/reference/api/kubeflow-pipeline-api-spec/#/definitions/v2beta1Run

```sh
python3 -m venv .venv
source .venv/bin/activate
```

## Curl Examples

```sh
export NAMESPACE_NAME="vllm-mistral-7b"
export PIPELINE_NAME="rag-ingestion"
# NAMESPACE_NAME="fraud-detection"
# export PIPELINE_NAME="fraud"
export SVC="https://ds-pipeline-dspa-${NAMESPACE_NAME}.apps.cluster-hxbsq.sandbox1380.opentlc.com"

export AUTH_HEADER="Bearer $(oc whoami -t)"
export METHOD="GET"
export CONTENT_TYPE="application/json"
export REQUEST_BODY="{}"

export PIPELINES=$(curl -s -X "${METHOD}" -H "Authorization: ${AUTH_HEADER}" ${SVC}/apis/v2beta1/pipelines)
echo $PIPELINES | jq .

PIPELINE=$(echo ${PIPELINES} | jq --arg name ${PIPELINE_NAME} '.pipelines[] | select(.display_name == $name)')
echo ${PIPELINE} | jq .
export PIPELINE_ID=$(echo ${PIPELINE} | jq -r .pipeline_id)
echo ${PIPELINE_ID}
```

```sh
CONTENT_TYPE="application/json"
REQUEST_BODY="{}"

# PIPELINE=$(curl -s -X "GET" -H "Authorization: ${AUTH_HEADER}" ${SVC}/apis/v2beta1/pipelines/${PIPELINE_ID} | jq)
# PIPELINE_NAME=$(echo ${PIPELINE} | jq -r .display_name)

PIPELINE_RUN=$(
curl -s -H "Content-Type: ${CONTENT_TYPE}" -H "Authorization: ${AUTH_HEADER}" -X POST ${SVC}/apis/v1beta1/runs \
-d @- << EOF
{
   "name":"${PIPELINE_NAME}_run",
   "pipeline_spec":{
      "pipeline_id":"${PIPELINE_ID}"
   }
}
EOF
)

PIPELINE_RUN=$(
curl -s -H "Content-Type: ${CONTENT_TYPE}" -H "Authorization: ${AUTH_HEADER}" -X POST ${SVC}/apis/v2beta1/runs \
-d @- << EOF
{
   "display_name":"${PIPELINE_NAME}_run",
   "description":"This is run from curl",
   "runtime_config": {
      "parameters": {}
    },
   "pipeline_version_reference":{
      "pipeline_id":"${PIPELINE_ID}"
   }
}
EOF
)

echo ${PIPELINE_RUN} | jq .

PIPELINE_RUN_ID=$(echo ${PIPELINE_RUN} | jq -r .run_id)
```

# ArgoCD Deployment

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  annotations:
    argocd.argoproj.io/compare-options: IgnoreExtraneous
  name: doc-bot
  namespace: openshift-gitops
spec:
  destination:
    namespace: vllm-mistral-7b
    server: 'https://kubernetes.default.svc'
  project: default
  source:
    helm:
      parameters:
        - name: instanceName
          value: vllm-mistral-7b
        - name: dataScienceProjectDisplayName
          value: vllm-mistral-7b
        - name: dataScienceProjectNamespace
          value: vllm-mistral-7b
    path: gitops/doc-bot
    repoURL: 'git@github.com:alpha-hack-program/doc-bot.git'
    targetRevision: main
  syncPolicy:
    automated:
      selfHeal: true
```