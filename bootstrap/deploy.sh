#!/bin/bash

# Load environment variables
. .env

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
  # annotations:
  #   argocd.argoproj.io/compare-options: IgnoreExtraneous
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
      parameters:
        - name: argocdNamespace
          value: "${ARGOCD_NAMESPACE}"
        - name: instanceName
          value: "vllm-mistral-7b"
        - name: dataScienceProjectDisplayName
          value: "Project ${DATA_SCIENCE_PROJECT_NAMESPACE}"
        - name: dataScienceProjectNamespace
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}"
        - name: milvus.name
          value: "milvus"
        - name: milvus.namespace
          value: "milvus"
        - name: milvus.username
          value: "root"
        - name: milvus.password
          value: "Milvus"
        - name: milvus.port
          value: "19530"
        - name: milvus.host
          value: "vectordb-milvus.milvus.svc.cluster.local"
        - name: chatApplication.name
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}-chat"
        - name: chatApplication.vcs.uri
          value: "https://github.com/alpha-hack-program/kb-chat.git"
        - name: chatApplication.vcs.ref
          value: "main"
        - name: chatApplication.vcs.name
          value: "alpha-hack-program/kb-chat"
        - name: chatApplication.vcs.path
          value: "kb-chat"
        - name: pipelinesApplication.name
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}-pipelines"
        - name: modelApplication.name
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}-mistral-7b"
        - name: modelConnection.awsAccessKeyId
          value: ${MINIO_ACCESS_KEY}
        - name: modelConnection.awsSecretAccessKey
          value: ${MINIO_SECRET_KEY}
        - name: modelConnection.awsS3Endpoint
          value: ${MINIO_ENDPOINT}
        - name: documentsConnection.awsAccessKeyId
          value: ${MINIO_ACCESS_KEY}
        - name: documentsConnection.awsSecretAccessKey
          value: ${MINIO_SECRET_KEY}
        - name: documentsConnection.awsS3Endpoint
          value: ${MINIO_ENDPOINT}
        - name: pipelinesConnection.awsAccessKeyId
          value: ${MINIO_ACCESS_KEY}
        - name: pipelinesConnection.awsSecretAccessKey
          value: ${MINIO_SECRET_KEY}
        - name: pipelinesConnection.awsS3Endpoint
          value: ${MINIO_ENDPOINT}
  syncPolicy:
    automated:
      # prune: true
      selfHeal: true
  ignoreDifferences:
    - group: apps
      kind: Deployment
      name: doc-bot
      jqPathExpressions:
        - '.spec.template.spec.containers[].image'
      
EOF

