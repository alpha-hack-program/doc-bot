Gradio example deploying an custom App

# Application

Gradio provides an API to deploy an instant url demo for applications. 

When we're using Gradio on OpenShift we need to know how many url we need to expose as we need run it as isolated container. 

# Lifecycle

```bash
podman build -t gradio-app .
```

```bash
podman run -p 7860:7860 gradio-app
```