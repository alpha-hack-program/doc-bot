import sys
import json
from regex import P
from transformers import AutoTokenizer, AutoModel
import torch

from pymilvus import MilvusClient
from pymilvus import connections


# Check that the number of arguments is correct
if len(sys.argv) != 4:
    print("Usage: python query.py <collection_name> <dossier> <query_text>")
    sys.exit(1)

# Get collection name and query text from arguments
collection_name = sys.argv[1]
dossier = sys.argv[2]
query_text = sys.argv[3]

def generate_embedding(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Connect to Milvus

# import requests

# # creating Session object and
# # declaring the verify variable to False
# session = requests.Session()
# session.verify = False

# # Load the self-signed certificate
# with open("/Users/cvicensa/Projects/openshift/alpha-hack-program/doc-bot/examples/ui/gradio-rag-milvus-vllm-openai/.venv/lib/python3.11/site-packages/certifi/cacert.pem", "rb") as f:
#     server_cert = f.read()

# # Establish a secure connection
# connections.connect(
#     alias="default",
#     uri="https://milvus-ia-sa.apps.ocp-ia.jccm.es",
#     user="root",
#     password="",
#     secure=True,
#     ca_pem=server_cert,
# )

# print("Connected to Milvus")

# client = MilvusClient(uri="tcp://localhost:19530", user="root", password="Milvus")
client = MilvusClient(uri="tcp://localhost:19530", user="root", password="")
# client = MilvusClient(uri="http://localhost:19530", user="root", password="Milvus")
# client = MilvusClient(uri="https://milvus-ia-sa.apps.ocp-ia.jccm.es", user="root", password="", ca_pem=server_cert,)

# Initialize Hugging Face model and tokenizer
# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_name = 'nomic-ai/nomic-embed-text-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_kwargs = {'trust_remote_code': True}
model = AutoModel.from_pretrained(model_name, trust_remote_code=True) # TODO: Add trust_remote_code=True to avoid warnings
vector_size = model.config.hidden_size

# Query vector
query_vector = generate_embedding(query_text)

# Search
k = 20
filter = f"dossier == '{dossier}'"
print(f"Searching for dossier: {dossier}...")
print(f"Query text: {query_text}")
print(f"Filter: {filter}")
search_res = client.search(
    collection_name=collection_name,
    data=[
        query_vector
    ],
    filter=filter,
    limit=k,  # Return top k results
    # search_params={"metric_type": "L2", "params": {"efConstruction": 200, "M": 16}}, # Euclidean distance
    search_params={"metric_type": "L2", "params": {"ef": 10}},  # Euclidean distance
    output_fields=["text", "source", "dossier"],
)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["entity"]["source"], res["distance"]) for res in search_res[0]
]

print("=======================================================")
print(f"Query text: {query_text}")
print(f"Retrieved {len(retrieved_lines_with_distances)} lines.")
print(json.dumps(retrieved_lines_with_distances, indent=4))


