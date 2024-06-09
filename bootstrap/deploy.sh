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
        - name: test
          value: "false"
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF

# Create an ArgoCD application to deploy the helm chart at this repository and path ./gitops/doc-bot
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: doc-bot
  namespace: openshift-gitops
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
        - name: test
          value: "false"
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF


# apiVersion: argoproj.io/v1alpha1
# kind: Application
# metadata:
#   name: {{ .Values.application.name }}
#   namespace: {{ .Values.application.namespace }}
# spec:
#   project: {{ .Values.application.project }}
#   source:
#     repoURL: {{ .Values.application.source.repoURL }}
#     path: {{ .Values.application.source.path }}
#     targetRevision: {{ .Values.application.source.targetRevision }}
#   destination:
#     server: {{ .Values.application.destination.server }}
#     namespace: {{ .Values.application.destination.namespace }}
#   syncPolicy:
#     {{- if .Values.application.syncPolicy.automated }}
#     automated: {}
#     {{- end }}
