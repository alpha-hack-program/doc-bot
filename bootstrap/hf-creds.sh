#!/bin/sh

# Load environment variables
. .env

# Check if namespace exists
if ! kubectl get namespace ${DATA_SCIENCE_PROJECT_NAMESPACE} > /dev/null 2>&1; then
  echo "Namespace ${DATA_SCIENCE_PROJECT_NAMESPACE} does not exist"
  exit 1
fi

# Check if .hf-creds file exists and if not, exit with an error
if [ ! -f .hf-creds ]; then
  echo "Please create a .hf-creds file with the Hugging Face API key"
  exit 1
fi

# Load creds from .hf-creds file
. .hf-creds

# Check if HF_USERNAME and HF_TOKEN are set
if [ -z "${HF_USERNAME}" ] || [ -z "${HF_TOKEN}" ]; then
  echo "Please set the HF_USERNAME and HF_TOKEN variables in the .hf-creds file"
  exit 1
fi

# Create a secret called hf-creds in the namespace ${DATA_SCIENCE_PROJECT_NAMESPACE}
kubectl create secret generic hf-creds \
  --from-literal=HF_USERNAME=${HF_USERNAME} \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${DATA_SCIENCE_PROJECT_NAMESPACE}

