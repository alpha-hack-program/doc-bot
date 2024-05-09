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

# Steps

## Model Serving

Deploy vLLM Model Serving instance in the OpenAI compatible API mode, either:

- as a custom server runtime in ODH/RHOAI.
- as a standalone server in OpenShift.

## Deploying the model

In both cases, deploy the model `mistralai/Mistral-7B-Instruct-v0.2`. The model can be found [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).



# QA

- How do you know how many GPU is consumed?

# Documentation

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/openshift/23.9.2/index.html)
- [How to install vLLM as a RHOAI runtime](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/serving-runtimes/vllm_runtime/README.md)
- [How to install vLLM as deployment](https://github.com/rh-aiservices-bu/llm-on-openshift/blob/main/llm-servers/vllm/README.md)
