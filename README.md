# Doc Bot Hacking Sprint

<table>
    <tr>
        <td><b>Title:</b></td>
        <td>Doc Bot</td>
    </tr>
    <tr>
        <td><b>Goal:</b></td>
        <td>Deploy RHOAI LLM <a href="https://ai-on-openshift.io/demos/llm-chat-doc/llm-chat-doc/#rag-chatbot-full-walkthrough">tutorial</a> and learn the inner workings of the system.</td>
    </tr>
    <tr>
        <td><b>Output:</b></td>
        <td>Being able to feed and deploy a RAG based chatbot or at least the backend/API of it.</td>
    </tr>
    <tr>
        <td><b>Timing:</b></td>
        <td>2 to 3h</td>
    </tr>
    <tr>
        <td><b>Notes:</b></td>
        <td>We'll start with the RHOAI Insurance Claim RHPDS demo.</td>
    </tr>
</table>

# Starting point of this hacking sprint

[Chat with your documentation](https://ai-on-openshift.io/demos/llm-chat-doc/llm-chat-doc/) lab from [AI on OpenShift](https://ai-on-openshift.io/).

# Objective

This repository exemplyfies the RAG pattern as a simple chatbot that can answer questions related to specific `dossiers`.

If you follow the installation instructions you should be able to deploy a full RAG system that can chunk and digest PDF documents related to `dossiers`, store them as vectors in Milvus and answer questions related to those documents (by dossier id) using an LLM and the context extracted runing the question through the vector database.

# Components

- Milvus as the vector database
- Mistral 7B run using the Model Serving capability of Red Hat OpenShift AI
- vLLM as the Serving Runtime implementing the Model Serving capability
- Document digestion Kubeflow pipeline run using the Red Hat OpenShift AI pipelines capability

# Installation

## Preparation

> If you need to delete a previous Red Hat OpenShift AI installation go [here](./docs/Uninstall.md)

Install the following operators prior to installing the OpenShift AI operator:

- OpenShift Serverless
- OpenShift Service Mesh

Check if the default storage class is set, if that is not the case you could use this command to set it up:

> **CAVEAT:** This is for an ocs enabled environment as the one you get from RHPDS AI demos like Parasol.

```sh
oc patch storageclass ocs-storagecluster-ceph-rbd -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

This demo expands on the environment of the Parasol demo, but it only relies on the Minio instance in the `ic-shared-minio` namespace.

Some manual steps:

  > Expand the MinIO PVC `minio` in namespace `ic-shared-minio` to 150Gi

  > Find the `bootstrap` ApplicationSet and delete all the elements in the list generator but `ic-shared-minio-app`.
  ```yaml
  generators:
    - list:
        elements:
          - cluster: in-cluster
            name: ic-shared-minio-app
            path: bootstrap/ic-shared-minio
            repoURL: 'https://github.com/rh-aiservices-bu/parasol-insurance.git'
            targetRevision: main
  ```

## Installing the Red Hat OpenShift AI operator

If you want to use stable versions use `stable`.

```sh
cat << EOF| oc create -f -
---
apiVersion: v1
kind: Namespace
metadata:
  name: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator 
spec:
  name: rhods-operator
  channel: stable-2.16
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Manual
EOF
```

If, on the other hand, you want to use the latest versions use `fast`.

```sh
cat << EOF| oc create -f -
---
apiVersion: v1
kind: Namespace
metadata:
  name: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: rhods-operator
  namespace: redhat-ods-operator 
spec:
  name: rhods-operator
  channel: fast
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Manual
  startingCSV: rhods-operator.2.15.0
EOF
```

The previous step creates a `manual` subscription that generates an `InstallPlan` that has to be approved.

> **CAUTION:** There should be only one install plan! If there are two delete one of them...

```sh
# List all install plans in namespace redhat-ods-operator then delete all but the first one
for installplan in $(oc get installplans -n redhat-ods-operator -o name); do
  # Check if the `INSTALL_PLAN_TO_PATCH` environment variable is empty.
  if [ -z "$INSTALL_PLAN_TO_PATCH" ]; then
    INSTALL_PLAN_TO_PATCH=$installplan
    echo "Approving $installplan"
    oc patch $installplan -n redhat-ods-operator --type merge --patch '{"spec":{"approved":true}}'
    continue
  fi
  echo "Deleting $installplan"
  oc delete $installplan -n redhat-ods-operator
done
```

Check if the plan was approved

```sh
echo "Install plan is approved? $(oc get installplans -n redhat-ods-operator -o jsonpath='{.items[].spec.approved}')"
```

## Create the Data Science Cluster

The Data Science Cluster object or `DSC` instructs the RHOAI operator how to install its compoments. In this case we will leave the default configuration for all the compoments except for `Kserve` this is because the default configuration uses a self signed certificate and we don't want that for our deployment.

There are three ways you can configure the certificate for `Kserve`:
- self-signed certificate, this is the default behaviour
- use the same certificate as the OpenShift routes in your cluster
- use other certificate, usually signed by a recognized CA

### Using the OpenShift routes certificate

Use this command to create the DSC and use the same certificate your OpenShift cluster is using.

```sh
cat << EOF| oc create -f -
---
apiVersion: datasciencecluster.opendatahub.io/v1
kind: DataScienceCluster
metadata:
  name: default-dsc
  labels:
    app.kubernetes.io/created-by: rhods-operator
    app.kubernetes.io/instance: default-dsc
    app.kubernetes.io/managed-by: kustomize
    app.kubernetes.io/name: datasciencecluster
    app.kubernetes.io/part-of: rhods-operator
spec:
  components:
    codeflare:
      managementState: Managed
    kserve:
      serving:
        ingressGateway:
          certificate:
            type: OpenshiftDefaultIngress
        managementState: Managed
        name: knative-serving
      managementState: Managed
    modelregistry:
      registriesNamespace: rhoai-model-registries
      managementState: Managed
    trustyai:
      managementState: Managed
    ray:
      managementState: Managed
    kueue:
      managementState: Managed
    workbenches:
      managementState: Managed
    dashboard:
      managementState: Managed
    modelmeshserving:
      managementState: Managed
    datasciencepipelines:
      managementState: Managed
    trainingoperator:
      managementState: Removed
EOF
```

### Using another certificate

In this case and for the sake of simplicity we are going to use the same certificate OpenShift is using but manually.

> It may sound silly to use the same certificate you get with the previous step but exemplifies the procedure to use whichever certificate you may need.

> Intructions extracted from: https://ai-on-openshift.io/odh-rhoai/single-stack-serving-certificate/

Get the name of the secret used for routing tasks in your OpenShift cluster:

```sh
export INGRESS_SECRET_NAME=$(oc get ingresscontroller default -n openshift-ingress-operator -o json | jq -r .spec.defaultCertificate.name)

oc get secret ${INGRESS_SECRET_NAME} -n openshift-ingress -o yaml > rhods-internal-primary-cert-bundle-secret.yaml
```

Clean `rhods-internal-primary-cert-bundle-secret.yaml`, change name to `rhods-internal-primary-cert-bundle-secret` and type to `kubernetes.io/tls`, it should look like:

```yaml
kind: Secret
apiVersion: v1
metadata:
name: rhods-internal-primary-cert-bundle-secret
data:
  tls.crt: >-
      LS0tLS1CRUd...
  tls.key: >-
      LS0tLS1CRUd...
type: kubernetes.io/tls
```

Create the secret in `istio-system`:

```sh
oc apply -n istio-system -f rhods-internal-primary-cert-bundle-secret.yaml
```

Check the secret is in place:

```sh
oc get secret rhods-internal-primary-cert-bundle-secret -n istio-system
```

Finally create the `DSC` object using the secret you just created.

```sh
cat << EOF| oc create -f -
---
kind: DataScienceCluster
apiVersion: datasciencecluster.opendatahub.io/v1
metadata:
  name: default-dsc
  labels:
    app.kubernetes.io/name: datasciencecluster
    app.kubernetes.io/instance: default-dsc
    app.kubernetes.io/part-of: rhods-operator
    app.kubernetes.io/managed-by: kustomize
    app.kubernetes.io/created-by: rhods-operator
spec:
  components:
    codeflare:
      managementState: Managed
    dashboard:
      managementState: Managed
    datasciencepipelines:
      managementState: Managed
    kserve:
      managementState: Managed
      serving:
        ingressGateway:
          certificate:
            # Intructions from: https://ai-on-openshift.io/odh-rhoai/single-stack-serving-certificate/
            # You have to copy the secret create it in istio-system namespace
            secretName: rhods-internal-primary-cert-bundle-secret
            type: Provided
        managementState: Managed
        name: knative-serving
    modelmeshserving:
      managementState: Managed
    kueue:
      managementState: Managed
    ray:
      managementState: Managed
    workbenches:
      managementState: Managed
EOF
```

Check if the `default-dsc` cluster is available. This command should return `True`.

```sh
oc get dsc default-dsc -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' && echo
```

# Doc Bot Application deployment

## Fork this reposiroty and clone it

Why forking? Because this demonstration tries to keep (to a certain extend) close to a real situation where repositories are private or password protected.

Clone your repo and change dir to the clone.

```sh
git clone <REPO_URL>
```

### Set some basic environment variables

Create a `.env` file in `bootstrap` or adapt the one provided if needed:

> This variables will be used by some scripts later. Pay special attention to **${DATA_SCIENCE_PROJECT_NAMESPACE}**.

> `GIT_BASE` and `REPO_URL` should point to your clone repo and base url.

```sh
GIT_BASE="https://github.com"
REPO_URL="https://github.com/alpha-hack-program/doc-bot.git"
MILVUS_NAMESPACE="milvus"
DATA_SCIENCE_PROJECT_NAMESPACE="doc-bot"
```

## Create a PAT for your forked repo

> You need this **ONLY** if your repo is private

You will have to provide a username and PAT to the script `create-secrets.sh` located at `bootstrap`.

Create a PAT and save it for later.

## Hugging Face prework

This demonstration will automatically download the LLM model files from Hugging Face (Mistral 7B by default) in order to do so you have to do sign up and generate a token and create a secret that will be used by a `Job`.

In order to simplify the creation of the secrete create a `.hf-creds` file in `bootstrap`.

```sh
HF_USERNAME=<USERNAME>
HF_TOKEN=hf_...
```

## Deploy Milvus and the Doc Bot Application using ArgoCD

Change dir to `bootstrap`:

```sh
cd bootstrap
```

Run the `deploy.sh` script from `bootstrap` folder:

```sh
./deploy.sh
```

## Create the git credentials secret
> **NOTE:** This is needed only if the git repository is private, for instance if you have forked this repository and made it private.

Create a PAT for the Tekton pipeline `kfp-upsert-pl` and another one for the build config `kb-chat` or one for both of them. This PAT should have read-only privileges on the git repository.

Run `create-secrets.sh` and be ready to input the username and PAT when asked for:

> **Only if needed**, remember.

```sh
./create-secrets.sh
```

## Create the Hugging Face secret

Run this script that uses the `.hf-creds` file you should have created in `bootstrap`:

From `bootstrap` run:

```sh
./hf-creds.sh
```

# Demo

Use the following command to get the URL of the chat to start chatting with your docs. From `bootstrap`:

```sh
. .env
oc get route kb-chat -n ${DATA_SCIENCE_PROJECT_NAMESPACE} -o jsonpath='{"https://"}{.spec.host}{"\n"}'
```



