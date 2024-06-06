#!/bin/bash

### Define Instance Types
INSTANCE_KEYS=(
  "Tesla T4 Single GPU"
  "Tesla T4 Multi GPU"
  "A10G Single GPU"
  "A10G Multi GPU"
  "A100"
  "H100"
  "DL1"
  "L4 Single GPU"
  "L4 Multi GPU"
)

INSTANCE_VALUES=(
  "g4dn.2xlarge"
  "g4dn.12xlarge"
  "g5.2xlarge"
  "g5.48xlarge"
  "p4d.24xlarge"
  "p5.48xlarge"
  "dl1.24xlarge"
  "g6.2xlarge"
  "g6.12xlarge"
)

# Function to get instance type by key
get_instance_type() {
  local key="$1"
  for i in "${!INSTANCE_KEYS[@]}"; do
    if [[ "${INSTANCE_KEYS[$i]}" == "$key" ]]; then
      echo "${INSTANCE_VALUES[$i]}"
      return 0
    fi
  done
  return 1  # Key not found
}

### Prompt User for GPU Instance Type
echo "### Select the GPU instance type:"
PS3='Please enter your choice: '
select opt in "${INSTANCE_KEYS[@]}"
do
  if [[ -n "$opt" ]]; then
    INSTANCE_TYPE=$(get_instance_type "$opt")
    if [[ $? -eq 0 ]]; then
      echo "### Selected GPU instance type: $opt"
      break
    else
      echo "--- Invalid option $REPLY ---"
    fi
  else
    echo "--- Invalid option $REPLY ---"
  fi
done

echo "### Selected GPU instance type: $INSTANCE_TYPE"


### Prompt User for Region
read -p "### Enter the AWS region (default: us-west-2): " REGION
REGION=${REGION:-us-west-2}

### Prompt User for Availability Zone
echo "### Select the availability zone (az1, az2, az3):"
PS3='Please enter your choice: '
az_options=("az1" "az2" "az3")
select az_opt in "${az_options[@]}"
do
  case $az_opt in
    "az1")
      AZ="${REGION}a"
      break
      ;;
    "az2")
      AZ="${REGION}b"
      break
      ;;
    "az3")
      AZ="${REGION}c"
      break
      ;;
    *) echo "--- Invalid option $REPLY ---";;
  esac
done

# Assign new name for the machineset
NEW_NAME="worker-gpu-$INSTANCE_TYPE-$AZ"

# Check if the machineset already exists
EXISTING_MACHINESET=$(oc get -n openshift-machine-api machinesets -o name | grep "$NEW_NAME")

if [ -n "$EXISTING_MACHINESET" ]; then
  echo "### Machineset $NEW_NAME already exists. Scaling to 1."
  oc scale --replicas=1 -n openshift-machine-api "$EXISTING_MACHINESET"
  echo "--- Machineset $NEW_NAME scaled to 1."
else
  echo "### Creating new machineset $NEW_NAME."
  oc get -n openshift-machine-api machinesets -o name | grep -v ocs | while read -r MACHINESET
  do
    oc get -n openshift-machine-api "$MACHINESET" -o json | jq '
        del( .metadata.uid, .metadata.managedFields, .metadata.selfLink, .metadata.resourceVersion, .metadata.creationTimestamp, .metadata.generation, .status) |
        (.metadata.name, .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"], .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]) |= sub("worker";"workerocs") |
        (.spec.template.spec.providerSpec.value.instanceType) |= "'"$INSTANCE_TYPE"'" |
        (.metadata.name) |= "'"$NEW_NAME"'" |
        (.spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]) |= "'"$NEW_NAME"'" |
        (.spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"]) |= "'"$NEW_NAME"'" |
        (.spec.template.spec.metadata.labels["node-role.kubernetes.io/gpu"]) |= "" |
        (.spec.template.spec.metadata.labels["cluster.ocs.openshift.io/openshift-storage"]) |= "" |
        (.spec.template.spec.taints) |= [{ "effect": "NoSchedule", "key": "nvidia.com/gpu" }]' | oc create -f -
    break
  done
  echo "--- New machineset $NEW_NAME created."
fi