# Pipelines KFP

https://www.kubeflow.org/docs/components/pipelines/v2/reference/api/kubeflow-pipeline-api-spec/#/definitions/v2beta1Run

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Curl Examples

```sh
export NAMESPACE_NAME="fraud-detection-ds"
export PIPELINE_NAME="deploy-model"

export SVC_HOST=$(oc get route ds-pipeline-dspa -n $NAMESPACE_NAME -o jsonpath='{.spec.host}')
export SVC="https://${SVC_HOST}"

export AUTH_HEADER="Bearer $(oc whoami -t)"

curl -X POST \
  -H "Authorization: ${AUTH_HEADER}" \
  -F "uploadfile=@./deploy.yaml" \
  ${SVC}/apis/v2beta1/pipelines/upload

```

```sh

export METHOD="GET"
export CONTENT_TYPE="application/json"
export REQUEST_BODY="{}"

export PIPELINES=$(curl -s -X "${METHOD}" -H "Authorization: ${AUTH_HEADER}" ${SVC}/apis/v2beta1/pipelines)
echo $PIPELINES | jq .

PIPELINE=$(echo ${PIPELINES} | jq --arg name ${PIPELINE_NAME} '.pipelines[] | select(.display_name == $name)')
echo ${PIPELINE} | jq .
export PIPELINE_ID=$(echo ${PIPELINE} | jq -r .pipeline_id)
echo ${PIPELINE_ID}
```

```sh
CONTENT_TYPE="application/json"
REQUEST_BODY="{}"

# PIPELINE=$(curl -s -X "GET" -H "Authorization: ${AUTH_HEADER}" ${SVC}/apis/v2beta1/pipelines/${PIPELINE_ID} | jq)
# PIPELINE_NAME=$(echo ${PIPELINE} | jq -r .display_name)

PIPELINE_RUN=$(
curl -s -H "Content-Type: ${CONTENT_TYPE}" -H "Authorization: ${AUTH_HEADER}" -X POST ${SVC}/apis/v1beta1/runs \
-d @- << EOF
{
   "name":"${PIPELINE_NAME}_run",
   "pipeline_spec":{
      "pipeline_id":"${PIPELINE_ID}"
   }
}
EOF
)

PIPELINE_RUN=$(
curl -s -H "Content-Type: ${CONTENT_TYPE}" -H "Authorization: ${AUTH_HEADER}" -X POST ${SVC}/apis/v2beta1/runs \
-d @- << EOF
{
   "display_name":"${PIPELINE_NAME}_run",
   "description":"This is run from curl",
   "runtime_config": {
      "parameters": {}
    },
   "pipeline_version_reference":{
      "pipeline_id":"${PIPELINE_ID}"
   }
}
EOF
)

echo ${PIPELINE_RUN} | jq .

PIPELINE_RUN_ID=$(echo ${PIPELINE_RUN} | jq -r .run_id)
```

