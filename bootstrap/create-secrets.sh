#!/bin/sh

. .env

# Environment variables
GIT_PAT_SECRET_NAME="git-pat-secret"
GIT_BASE="https://gitlab.cm.jccm.es"
# GIT_URL="${GIT_BASE}/inteligencia-artificial/geneva/geneva-ocp.git"
GIT_USERNAME="gitlab"

NAMESPACE_STATUS=$(kubectl get namespace/${DATA_SCIENCE_PROJECT_NAMESPACE} -o jsonpath='{.status.phase}')
if [ "${NAMESPACE_STATUS}" == *"Active"* ]; then
    echo "Wait until ArgoCD has create ns ${DATA_SCIENCE_PROJECT_NAMESPACE} or create it manually"
    exit 1
fi

echo "PAT for ${GIT_BASE}: " && read -s GIT_PAT
if [ -z "${GIT_BASE}" ]; then
    echo "You should provide a PAT for ${GIT_BASE}"
    exit 1
fi

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ${GIT_PAT_SECRET_NAME}
  namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
type: kubernetes.io/basic-auth
stringData:
  user.name: ${GIT_USERNAME}
  user.email: "${GIT_USERNAME}@example.com"
  username: ${GIT_USERNAME}
  password: ${GIT_PAT}
EOF

kubectl annotate --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${GIT_PAT_SECRET_NAME} \
  "tekton.dev/git-0=${GIT_BASE}"

kubectl annotate --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${GIT_PAT_SECRET_NAME} \
  "build.openshift.io/source-secret-match-uri-1=${GIT_BASE}/*"

kubectl label --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${GIT_PAT_SECRET_NAME} \
  credential.sync.jenkins.openshift.io=true
