#!/bin/bash

# Load environment variables
. .env

# Check if the required environment variables are set
if [ -z "$ARGOCD_NAMESPACE" ] || [ -z "$MILVUS_NAMESPACE" ] || [ -z "$REPO_URL" ] || [ -z "$DATA_SCIENCE_PROJECT_NAMESPACE" ]; then
  echo "Error: Required environment variables are not set."
  exit 1
fi
if [ -z "$MINIO_ACCESS_KEY" ] || [ -z "$MINIO_SECRET_KEY" ] || [ -z "$MINIO_ENDPOINT" ]; then
  echo "Error: MinIO credentials are not set."
  exit 1
fi
if [ -z "$GPU_NAME" ]; then
  echo "Error: GPU name is not set."
  exit 1
fi

# Check if hf-creds.sh exists
if [ ! -f "./hf-creds.sh" ]; then
  echo "Error: hf-creds.sh not found."
  exit 1
fi

# If CHAT_APPLICATION_LANGUAGE is not set, set it to "en"
if [ -z "$CHAT_APPLICATION_LANGUAGE" ]; then
  CHAT_APPLICATION_LANGUAGE="en"
fi

echo "DATA_SCIENCE_PROJECT_NAMESPACE: ${DATA_SCIENCE_PROJECT_NAMESPACE}"

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/milvus
cat <<EOF | oc apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: milvus
  namespace: ${ARGOCD_NAMESPACE}
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: ${MILVUS_NAMESPACE}
  source:
    path: gitops/milvus
    repoURL: ${REPO_URL}
    targetRevision: main
    helm:
      parameters:
        - name: milvusNamespace
          value: "${MILVUS_NAMESPACE}"
  syncPolicy:
    automated:
      # prune: true
      selfHeal: true
EOF

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/doc-bot
cat <<EOF | oc apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ${DATA_SCIENCE_PROJECT_NAMESPACE}
  namespace: ${ARGOCD_NAMESPACE}
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
  source:
    path: gitops/doc-bot
    repoURL: ${REPO_URL}
    targetRevision: ${GIT_REVISION:-main}
    helm:
      values: |
        argocdNamespace: "${ARGOCD_NAMESPACE}"
        instanceName: "vllm-mistral-7b"
        dataScienceProjectDisplayName: "Project ${DATA_SCIENCE_PROJECT_NAMESPACE}"
        dataScienceProjectNamespace: "${DATA_SCIENCE_PROJECT_NAMESPACE}"
        milvus:
          name: "milvus"
          namespace: "milvus"
          username: "root"
          password: "Milvus"
          port: 19530
          host: "vectordb-milvus.milvus.svc.cluster.local"
        chatApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-chat"
          vcs:
            uri: "https://github.com/alpha-hack-program/kb-chat.git"
            ref: "main"
            name: "alpha-hack-program/kb-chat"
            path: "kb-chat"
          language: "${CHAT_APPLICATION_LANGUAGE}"
        pipelinesApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-pipelines"
        modelApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-model"
          repoURL: "${MODEL_SERVING_REPO_URL:-https://github.com/alpha-hack-program/model-serving-utils.git}"
          targetRevision: "${MODEL_SERVING_TARGET_REVISION:-main}"
          model:
            connection:
              name: "llm"
              displayName: "llm"
              awsAccessKeyId: ${MINIO_ACCESS_KEY}
              awsSecretAccessKey: ${MINIO_SECRET_KEY}
              awsS3Endpoint: ${MINIO_ENDPOINT}
            accelerator:
              productName: ${GPU_NAME}
        embeddingsApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-embeddings"
          repoURL: "${MODEL_SERVING_REPO_URL:-https://github.com/alpha-hack-program/model-serving-utils.git}"
          targetRevision: "${MODEL_SERVING_TARGET_REVISION:-main}"
          model:
            connection:
              name: "embeddings"
              displayName: "embeddings"
              awsAccessKeyId: ${MINIO_ACCESS_KEY}
              awsSecretAccessKey: ${MINIO_SECRET_KEY}
              awsS3Endpoint: ${MINIO_ENDPOINT}
            accelerator:
              productName: ${GPU_NAME}
        kotaemonApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-kotaemon"
        model:
          connection:
            awsAccessKeyId: ${MINIO_ACCESS_KEY}
            awsSecretAccessKey: ${MINIO_SECRET_KEY}
            awsS3Endpoint: ${MINIO_ENDPOINT}
          accelerator:
            productName: ${GPU_NAME}
        embeddings:
          connection:
            awsAccessKeyId: ${MINIO_ACCESS_KEY}
            awsSecretAccessKey: ${MINIO_SECRET_KEY}
            awsS3Endpoint: ${MINIO_ENDPOINT}
        documentsConnection:
          awsAccessKeyId: ${MINIO_ACCESS_KEY}
          awsSecretAccessKey: ${MINIO_SECRET_KEY}
          awsS3Endpoint: ${MINIO_ENDPOINT}
        pipelinesConnection:
          awsAccessKeyId: ${MINIO_ACCESS_KEY}
          awsSecretAccessKey: ${MINIO_SECRET_KEY}
          awsS3Endpoint: ${MINIO_ENDPOINT}
  syncPolicy:
    automated:
      selfHeal: true
  ignoreDifferences:
    - group: apps
      kind: Deployment
      name: doc-bot
      jqPathExpressions:
        - '.spec.template.spec.containers[].image'
EOF

# Check if the namespace exists
if oc get namespace ${DATA_SCIENCE_PROJECT_NAMESPACE} >/dev/null 2>&1; then
  echo "Namespace ${DATA_SCIENCE_PROJECT_NAMESPACE} already exists."
else
  # Create the namespace
  oc create namespace ${DATA_SCIENCE_PROJECT_NAMESPACE}
fi

# Create HF credentials
./hf-creds.sh