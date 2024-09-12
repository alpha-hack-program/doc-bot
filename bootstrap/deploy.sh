#!/bin/bash

# Load environment variables
. .env

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/milvus
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: milvus
  namespace: openshift-gitops
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
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: doc-bot
  namespace: openshift-gitops
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
        - name: chatApplication.vcs.uri
          value: "https://github.com/alpha-hack-program/kb-chat.git"
        - name: chatApplication.vcs.ref
          value: "main"
        - name: chatApplication.vcs.name
          value: "alpha-hack-program/kb-chat"
        - name: chatApplication.vcs.path
          value: "kb-chat"
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

