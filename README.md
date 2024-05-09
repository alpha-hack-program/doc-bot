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

# Steps

## Model Serving

Deploy vLLM Model Serving instance in the OpenAI compatible API mode, either:

- [as a custom server runtime in ODH/RHOAI](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/serving-runtimes/vllm_runtime/README.md)
- [as a standalone server in OpenShift](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/llm-servers/vllm/README.md)

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

# QA

- How do you know how many GPU is consumed?

# Documentation

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/23.9.2/index.html)
- [How to install vLLM as a RHOAI runtime](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/serving-runtimes/vllm_runtime/README.md)
- [How to install vLLM as deployment](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/llm-servers/vllm/README.md)
