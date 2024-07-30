from ast import If
import json
import os
import random
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional, List, Dict, Any

from kubernetes import client, config

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLMOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from milvus_retriever_with_score_threshold import MilvusRetrieverWithScoreThreshold

load_dotenv()

# Load in-cluster Kubernetes configuration but if it fails, load local configuration
try:
    config.load_incluster_config()
except config.config_exception.ConfigException:
    config.load_kube_config()

# Get prediction URL by name and namespace
def get_predictor_url(namespace="default", predictor_name="mistral-7b-predictor"):
    api_instance = client.CustomObjectsApi()
    try:
        predictor = api_instance.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=predictor_name
        )
        return f"{predictor['status']['url']}"
    except Exception as e:
        print(f"Error retrieving predictor {predictor_name} in namespace {namespace}: {e}")
        return None

# Get NAMESPACE from environment
NAMESPACE = os.getenv('NAMESPACE')
if not NAMESPACE:
    # Get the current namespace or error if not found
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            NAMESPACE = f.read().strip()
    except FileNotFoundError:
        raise ValueError("NAMESPACE environment variable not set and could not get current namespace.")

# Get PREDICTOR_NAME from environment
PREDICTOR_NAME = os.getenv('PREDICTOR_NAME')
if not PREDICTOR_NAME:
    raise ValueError("PREDICTOR_NAME environment variable not set.")

# Get INFERENCE_SERVER_URL from environment
INFERENCE_SERVER_URL = os.getenv('INFERENCE_SERVER_URL')
if not INFERENCE_SERVER_URL:
    predictor_url = get_predictor_url(namespace=NAMESPACE, predictor_name=PREDICTOR_NAME)
    if predictor_url:
        INFERENCE_SERVER_URL = predictor_url
    else:
        raise ValueError("INFERENCE_SERVER_URL environment variable not set.")

# Parameters

APP_TITLE = os.getenv('APP_TITLE', 'Chat with your Knowledge Base!')
SHOW_TITLE_IMAGE = os.getenv('SHOW_TITLE_IMAGE', 'True')

MODEL_NAME = os.getenv('MODEL_NAME')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))
TOP_P = float(os.getenv('TOP_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
PRESENCE_PENALTY = float(os.getenv('PRESENCE_PENALTY', 1.03))

MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USERNAME = os.getenv('MILVUS_USERNAME')
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
MILVUS_COLLECTIONS_FILE = os.getenv('MILVUS_COLLECTIONS_FILE')

DEFAULT_COLLECTION = os.getenv('DEFAULT_COLLECTION')
DEFAULT_DOSSIER = 'None'
PROMPT_FILE = os.getenv('PROMPT_FILE', 'default_prompt.txt')
MAX_RETRIEVED_DOCS = int(os.getenv('MAX_RETRIEVED_DOCS', 4))
SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.99))

# Load collections from JSON file
with open(MILVUS_COLLECTIONS_FILE, 'r') as file:
    collections_data = json.load(file)

# Load Prompt template from txt file
with open(PROMPT_FILE, 'r') as file:
    prompt_template = file.read()

############################
# Streaming call functions #
############################
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_source_duplicates(input_list):
    unique_list = []
    for item in input_list:
        if item.metadata['source'] not in unique_list:
            unique_list.append(item.metadata['source'])
    return unique_list

def stream(input_text, selected_collection, selected_dossier) -> Generator:
    # A Queue is needed for Streaming implementation
    q = Queue()

    print("Starting streaming")
    print("Input text: ", input_text)
    print("Selected collection: ", selected_collection)
    print("Selected dossier: ", selected_dossier)
    print("INFERENCE_SERVER_URL: ", INFERENCE_SERVER_URL)
    print("MODEL_NAME: ", MODEL_NAME)

    # Instantiate LLM
    llm =  VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base=f'{INFERENCE_SERVER_URL}/v1',
        model_name=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=True,
        verbose=False,
        callbacks=[QueueCallback(q)]
    )

    # Instantiate QA chain
    retriever = MilvusRetrieverWithScoreThreshold(
        embedding_function=embeddings,
        collection_name=selected_collection,
        collection_description="",
        collection_properties=None,
        dossier_name=selected_dossier,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
        consistency_level="Session",
        search_params=None,
        k=MAX_RETRIEVED_DOCS,
        score_threshold=SCORE_THRESHOLD,
        # metadata_field="metadata",
        # text_field="page_content",
        text_field="text",
        vector_field="vector"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_chain_prompt},
        return_source_documents=True
        )

    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = qa_chain.invoke({"query": input_text})
        sources = remove_source_duplicates(resp['source_documents'])
        if len(sources) != 0:
            q.put("\n*Sources:* \n")
            for source in sources:
                q.put("* " + str(source) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

######################
# LLM chain elements #
######################

# Document store: Milvus
model_kwargs = {'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs=model_kwargs,
    show_progress=False
)

# Prompt
qa_chain_prompt = PromptTemplate.from_template(prompt_template)


# Dossier update function
dossier = None

def update_dossier(dossier_number):
    global dossier
    dossier = dossier_number

####################
# Gradio interface #
####################

collection_options = [(collection['display_name'], collection['name']) for collection in collections_data]
dossier_options = ['None', '3166', '3210']

def select_collection(collection_name, selected_collection):
    return {
        selected_collection_var: collection_name
        }

def select_dossier(dossier_name, selected_dossier):
    return {
        selected_dossier_var: dossier_name
        }

def ask_llm(message, history, selected_collection, selected_dossier):
    for next_token, content in stream(message, selected_collection, selected_dossier):
        yield(content)

css = """
footer {visibility: hidden}
.title_image img {width: 80px !important}
"""

with gr.Blocks(title="Knowledge base backed Chatbot", css=css) as demo:
    selected_collection_var = gr.State(DEFAULT_COLLECTION)
    selected_dossier_var = gr.State(DEFAULT_DOSSIER)
    with gr.Row():
        if SHOW_TITLE_IMAGE == 'True':
            gr.Markdown(f"# ![image](/file=./assets/reading-robot.png)   {APP_TITLE}")
        else:
            gr.Markdown(f"# {APP_TITLE}")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"This chatbot lets you chat with a Large Language Model (LLM) that can be backed by different knowledge bases (or none).")
            collection = gr.Dropdown(
                choices=collection_options,
                label="Knowledge Base:",
                value=DEFAULT_COLLECTION,
                interactive=True,
                info="Choose the knowledge base the LLM will have access to:"
            )
            collection.input(select_collection, inputs=[collection,selected_collection_var], outputs=[selected_collection_var])
            dossier = gr.Dropdown(
                choices=dossier_options,
                label="Select Dossier:",
                value=dossier_options[0],
                interactive=True,
                info="Choose a dossier:"
            )
            dossier.input(select_dossier, inputs=[dossier,selected_dossier_var], outputs=[selected_dossier_var])
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                show_label=False,
                avatar_images=(None,'assets/robot-head.svg'),
                render=False,
                show_copy_button=True
                )
            gr.ChatInterface(
                ask_llm,
                additional_inputs=[selected_collection_var, selected_dossier_var],
                chatbot=chatbot,
                clear_btn=None,
                retry_btn=None,
                undo_btn=None,
                stop_btn=None,
                description=None
                )

if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=10
        ).launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/robot-head.ico',
        allowed_paths=["./assets/"]
        )
