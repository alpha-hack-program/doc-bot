#!/bin/sh
ARGOCD_APP_NAME=vllm-mistral-7b

# Load environment variables
DATA_SCIENCE_PROJECT_NAMESPACE="vllm-mistral-7b-instruct-v0.2"

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set instanceName="vllm-mistral-7b" \
  --set dataScienceProjectNamespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set dataScienceProjectDisplayName=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set model.root=mistralai \
  --set model.id=Mistral-7B-Instruct-v0.2 \
  --set model.name=mistral-7b \
  --set model.displayName="Mistral 7b" \
  --set model.accelerator.productName="NVIDIA-A10G" \
  --set model.accelerator.min=1 \
  --set model.accelerator.max=1 \
  --include-crds