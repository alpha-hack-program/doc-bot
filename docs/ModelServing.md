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

