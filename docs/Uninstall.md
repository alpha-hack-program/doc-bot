# Procedure

https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html/installing_and_uninstalling_openshift_ai_self-managed/uninstalling-openshift-ai-self-managed_uninstalling-openshift-ai-self-managed

Open a new terminal window.
In the OpenShift command-line interface (CLI), log in to your OpenShift Container Platform cluster as a cluster administrator, as shown in the following example:

```sh
oc login <openshift_cluster_url> -u system:admin
```

Delete DCSs:

```sh
for dsc in $(oc get dsc | awk '{print $1}'); do
  oc delete dsc $dsc
done
```

Create a ConfigMap object for deletion of the Red Hat OpenShift AI Operator.

```sh
oc create configmap delete-self-managed-odh -n redhat-ods-operator
```

To delete the rhods-operator, set the addon-managed-odh-delete label to true.

```sh
oc label configmap/delete-self-managed-odh api.openshift.com/addon-managed-odh-delete=true -n redhat-ods-operator
```

When all objects associated with the Operator are removed, delete the redhat-ods-operator project.

Set an environment variable for the redhat-ods-applications project.

```sh
PROJECT_NAME=redhat-ods-applications
```

Wait until the redhat-ods-applications project has been deleted.

```sh
while oc get project $PROJECT_NAME &> /dev/null; do
echo "The $PROJECT_NAME project still exists"
sleep 1
done
echo "The $PROJECT_NAME project no longer exists"
```

When the redhat-ods-applications project has been deleted, you see the following output.

```
The redhat-ods-applications project no longer exists
```

When the redhat-ods-applications project has been deleted, delete the redhat-ods-operator project.

```sh
oc delete namespace redhat-ods-operator
```

**Verification**

Confirm that the rhods-operator subscription no longer exists.

```sh
oc get subscriptions --all-namespaces | grep rhods-operator
```

Confirm that the following projects no longer exist.

redhat-ods-applications
redhat-ods-monitoring
redhat-ods-operator
rhods-notebooks

```sh
for ns in $(oc get ns | grep -i redhat-ods | awk '{print $1}'); do
  oc delete ns $ns
done
for ns in $(oc get ns | grep -i rhods | awk '{print $1}'); do
  oc delete ns $ns
done
```

Check:

```sh
oc get namespaces | grep -i redhat-ods
oc get namespaces | grep -i rhods
```

Delete DCSs:

```sh
for dsc in $(oc get dsc | awk '{print $1}'); do
  oc delete dsc $dsc
done
```

Delete CRDs

> This usually doesn't do the all the work... and you have to delete the finalizer manually of **DSPAs**!!!. Remember this should be for demos starting with RHPDS not in a real enviroment where the installation doesn't involve necesarily unistalling RHOAI.

```sh
for crd in $(oc get crd | grep opendatahub.io | awk '{print $1}'); do
  oc delete crd $crd
done
```

# Unless it's needed and shared by other elements

**Delete Knative Serving**

```sh
oc delete knativeserving/knative-serving -n knative-serving
```

**Delete Knative Eventing**

```sh
oc delete knativeeventing/knative-eventing -n knative-eventing
```

**Delete Service Mesh Control Plane objects**

```sh
for smcp in $(oc get servicemeshcontrolplane -n istio-system -o name); do
  kubectl delete $smcp -n istio-system
done
```

And check if they're all gone.

```sh
oc get servicemeshcontrolplane -n istio-system -o name
```

Delete installplans of the operator if found:

```sh
for installplan in $(oc get installplans -n redhat-ods-operator -o name); do
  oc delete $installplan -n redhat-ods-operator
done
```