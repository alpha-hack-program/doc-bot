#!/bin/sh

. .env

if [ -z "${GIT_BASE}" ]; then
    echo "You should provide a git base url or add it to .env file"
    exit 1
fi

# Git username
echo "Username for ${GIT_BASE}: " && read GIT_USERNAME
if [ -z "${GIT_USERNAME}" ]; then
    echo "You should provide a username for ${GIT_BASE}"
    exit 1
fi

# Create a secret for BuildConfig
BC_GIT_PAT_SECRET_NAME="git-pat-bc-secret"

# Check if the namespace exists
NAMESPACE_STATUS=$(oc get namespace/${DATA_SCIENCE_PROJECT_NAMESPACE} -o jsonpath='{.status.phase}')
if [ "${NAMESPACE_STATUS}" == *"Active"* ]; then
    echo "Wait until ArgoCD has create ns ${DATA_SCIENCE_PROJECT_NAMESPACE} or create it manually"
    exit 1
fi

echo "PAT for BC for ${GIT_BASE}: " && read -s BC_GIT_PAT
if [ -z "${BC_GIT_PAT}" ]; then
    echo "You should provide a PAT for ${GIT_BASE}"
    exit 1
fi

cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ${BC_GIT_PAT_SECRET_NAME}
  namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
type: kubernetes.io/basic-auth
stringData:
  username: ${GIT_USERNAME}
  password: ${BC_GIT_PAT}
EOF

# Create a secret for Tekton
TK_GIT_PAT_SECRET_NAME="git-pat-tk-secret"

echo "PAT for Tekton for ${GIT_BASE}: " && read -s TK_GIT_PAT
if [ -z "${TK_GIT_PAT}" ]; then
    echo "You should provide a PAT for ${GIT_BASE}"
    exit 1
fi

cat <<EOF | oc apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ${TK_GIT_PAT_SECRET_NAME}
  namespace: ${DATA_SCIENCE_PROJECT_NAMESPACE}
type: kubernetes.io/basic-auth
stringData:
  user.name: ${GIT_USERNAME}
  user.email: "${GIT_USERNAME}@example.com"
  username: ${GIT_USERNAME}
  password: ${TK_GIT_PAT}
EOF

oc annotate --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${TK_GIT_PAT_SECRET_NAME} \
  "tekton.dev/git-0=${GIT_BASE}"

oc annotate --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${TK_GIT_PAT_SECRET_NAME} \
  "build.openshift.io/source-secret-match-uri-1=${GIT_BASE}/*"

oc label --overwrite -n ${DATA_SCIENCE_PROJECT_NAMESPACE} secret ${TK_GIT_PAT_SECRET_NAME} \
  "credential.sync.jenkins.openshift.io=true"
