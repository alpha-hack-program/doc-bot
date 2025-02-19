#!/bin/bash

# Load environment variables
. .env

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
    targetRevision: main
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
        embeddingsApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-embeddings"
        kotaemonApplication:
          name: "${DATA_SCIENCE_PROJECT_NAMESPACE}-kotaemon"
        model:
          connection:
            awsAccessKeyId: ${MINIO_ACCESS_KEY}
            awsSecretAccessKey: ${MINIO_SECRET_KEY}
            awsS3Endpoint: ${MINIO_ENDPOINT}
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

