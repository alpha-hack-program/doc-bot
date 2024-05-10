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

Once the stack is installed, adding the runtime is pretty straightforward:

- As an admin, in the OpenShift AI Dashboard, open the menu `Settings -> Serving runtimes`.
- Click on `Add serving runtime`.
- For the type of model serving platforms this runtime supports, select `Single model serving platform`.
- Upload the file `./vllm_runtime/vllm-runtime.yaml`, or click `Start from scratch` and copy/paste its content.

The runtime is now available when deploying a model.

### As a deployment [optional]

TODO

## Deploying the model

In both cases, deploy the model `mistralai/Mistral-7B-Instruct-v0.2`. The model can be found [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

> **NOTE:** https://huggingface.co/docs/hub/models-downloading

Some hints:
> [Click here to install `git lfs`](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 

Get a PAT from HF and put it in `.hf-token`.

```sh
export HF_USERNAME=cvicens
export HF_TOKEN=$(cat .hf-token)
export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
mkdir tmp
cd tmp
git lfs install
git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${MODEL_ID}
```

### Copy the model files to an S3 bucket

Upload to an S3 bucket:

```sh
export AWS_ACCESS_KEY_ID=your_minio_access_key
export AWS_SECRET_ACCESS_KEY=your_minio_secret_key
export AWS_DEFAULT_REGION=none  # Any value is fine
export AWS_S3_ENDPOINT=your_minio_endpoint  # e.g., http://localhost:9000
export AWS_S3_CUSTOM_DOMAIN=${AWS_S3_ENDPOINT}
export AWS_S3_USE_PATH_STYLE=1

aws configure set default.s3.endpoint_url ${AWS_S3_ENDPOINT}
aws configure set default.s3.addressing_style path
aws configure set default.s3.region ${AWS_DEFAULT_REGION}  # Any value is fine

aws s3 cp ${MODEL_ID}/ s3://mybucket/ --recursive 
```

### Deploy the model to vLLM

This runtime can be used in the exact same way as the out of the box ones:

- Create a connection to the S3 bucket that contains the model files.
- Deploy the model from the Dashboard.
- Make sure you have added a GPU to your GPU configuration, that you have enough VRAM (GPU memory) to load the model, and that you have enough standard memory (RAM). Although the model loads into the GPU, RAM is still used for the pre-loading operations.
- Once the model is loaded, you can access the inference endpoint provided through the dashboard.

### Usage

This implementation of the runtime provides an **OpenAI compatible API**. So any tool or library that can connect to OpenAI services will be able to consume the endpoint.

Python and Curl examples are provided [here](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-completions-api-with-vllm).

You can also find a notebook example using Langchain to query vLLM in this repo [here](../../examples/notebooks/langchain/Langchain-vLLM-Prompt-memory.ipynb).

Also, vLLM provides a full Swagger UI where you can get the full documentation of the API (methods, parameters), and try it directly without any coding,... It is accessible at the address `https://your-endpoint-address/docs`.

# Vector Store

## Milvus deployment

For our RAG we will need a Vector Database to store the Embeddings of the different documents. In this example we are using Milvus.

Deployment instructions specific to OpenShift are available [here](https://github.com/atarazana/doc-bot/tree/main/vector-databases/milvus).

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

> **REVISIT:** Apparently there is an operator but not ready as of the time of writing this document.

This deployment method is based on the [Offline installation](https://milvus.io/docs/install_offline-helm.md) that purely rely on Helm Charts.

- Log into your OpenShift cluster, and create a new project to host your Milvus installation:

```bash
oc new-project milvus
```

- Add and update Milvus Helm repository locally:

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update
```

- Fetch the file [`openshift-values.yaml`](openshift-values.yaml) from this repo. This file is really important as it sets specific values for OpenShift compatibility. You can also modify some of the values in this file to adapt the deployment to your requirements, notably modify the Minio admin user and password.

    ```bash
    wget https://raw.githubusercontent.com/atarazana/doc-bot/main/vector-databases/milvus/openshift-values.yaml
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

  - For Milvus Cluster:
  
    ```bash
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.securityContext) = {}' -i milvus_manifest_cluster.yaml
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-etcd") | .spec.template.spec.containers[0].securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_cluster.yaml
    yq '(select(.kind == "StatefulSet" and .metadata.name == "vectordb-minio") | .spec.template.spec.securityContext) = {"capabilities": {"drop": ["ALL"]}, "runAsNonRoot": true, "allowPrivilegeEscalation": false}' -i milvus_manifest_cluster.yaml
    ```

- Deploy Milvus (eventually change the name of the manifest)!

  - For Milvus Standalone:
  
    ```bash
    oc apply -f milvus_manifest_standalone.yaml
    ```

  - For Milvus Cluster:
  
    ```bash
    oc apply -f milvus_manifest_cluster.yaml
    ```

- To deploy the management UI for Milvus, called Attu, apply the file [attu-deployment.yaml](attu-deployment.yaml):

```bash
oc apply -f https://raw.githubusercontent.com/atarazana/doc-bot/main/vector-databases/milvus/attu-deployment.yaml
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


# QA

- How do you know how many GPU is consumed?

# Documentation

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/23.9.2/index.html)
- [How to install vLLM as a RHOAI runtime](https://github.com/atarazana/doc-bot/blob/main/serving-runtimes/vllm_runtime/README.md)
- [How to install vLLM as deployment](https://github.com/atarazana/doc-bot/blob/main/llm-servers/vllm/README.md)
