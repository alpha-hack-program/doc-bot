#!/bin/sh

# Load environment variables
. .env

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/doc-bot
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
  name: doc-bot-2x-t4
  namespace: openshift-gitops
  annotations:
    argocd.argoproj.io/compare-options: IgnoreExtraneous
spec:
  project: default
  destination:
    server: 'https://kubernetes.default.svc'
    namespace: "${DATA_SCIENCE_PROJECT_NAMESPACE}-2x-t4"
  source:
    path: gitops/doc-bot
    repoURL: ${REPO_URL}
    targetRevision: main
    helm:
      parameters:
        - name: instanceName
          value: "vllm-mistral-7b-2x-t4"
        - name: dataScienceProjectNamespace
          value: "${DATA_SCIENCE_PROJECT_NAMESPACE}-2x-t4"
        - name: dataScienceProjectDisplayName
          value: "Project ${DATA_SCIENCE_PROJECT_NAMESPACE}-2x-t4"
        - name: model.accelerator.productName
          value: "Tesla-T4"
        - name: model.accelerator.min
          value: "2"
        - name: model.accelerator.max
          value: "2"
  syncPolicy:
    automated:
      # prune: true
      selfHeal: true
EOF

