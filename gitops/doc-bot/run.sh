#!/bin/sh
ARGOCD_APP_NAME=doc-bot

# Load environment variables
. ../../bootstrap/.env

helm template . --name-template ${ARGOCD_APP_NAME} \
  --set argocdNamespace="${ARGOCD_NAMESPACE}" \
  --set instanceName="vllm-mistral-7b" \
  --set dataScienceProjectNamespace=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set dataScienceProjectDisplayName=${DATA_SCIENCE_PROJECT_NAMESPACE} \
  --set model.accelerator.productName="Tesla-T4" \
  --set model.accelerator.min=2 \
  --set model.accelerator.max=2 \
  --include-crds