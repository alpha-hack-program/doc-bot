#!/bin/sh
ARGOCD_APP_NAME=vllm-mistral-7b

# Load environment variables
DATA_SCIENCE_PROJECT_NAMESPACE="vllm-mistral-7b-instruct-v0.2"

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set namespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --include-crds