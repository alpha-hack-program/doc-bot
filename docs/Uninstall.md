Procedure

https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/2.8/html/installing_and_uninstalling_openshift_ai_self-managed/uninstalling-openshift-ai-self-managed_uninstalling-openshift-ai-self-managed

Open a new terminal window.
In the OpenShift command-line interface (CLI), log in to your OpenShift Container Platform cluster as a cluster administrator, as shown in the following example:

$ oc login <openshift_cluster_url> -u system:admin
Create a ConfigMap object for deletion of the Red Hat OpenShift AI Operator.

$ oc create configmap delete-self-managed-odh -n redhat-ods-operator
To delete the rhods-operator, set the addon-managed-odh-delete label to true.

$ oc label configmap/delete-self-managed-odh api.openshift.com/addon-managed-odh-delete=true -n redhat-ods-operator
When all objects associated with the Operator are removed, delete the redhat-ods-operator project.

Set an environment variable for the redhat-ods-applications project.

$ PROJECT_NAME=redhat-ods-applications
Wait until the redhat-ods-applications project has been deleted.

$ while oc get project $PROJECT_NAME &> /dev/null; do
echo "The $PROJECT_NAME project still exists"
sleep 1
done
echo "The $PROJECT_NAME project no longer exists"
When the redhat-ods-applications project has been deleted, you see the following output.

The redhat-ods-applications project no longer exists
When the redhat-ods-applications project has been deleted, delete the redhat-ods-operator project.

$ oc delete namespace redhat-ods-operator
Verification

Confirm that the rhods-operator subscription no longer exists.

$ oc get subscriptions --all-namespaces | grep rhods-operator
Confirm that the following projects no longer exist.

redhat-ods-applications
redhat-ods-monitoring
redhat-ods-operator
rhods-notebooks

$ oc get namespaces | grep -e redhat-ods* -e rhods*