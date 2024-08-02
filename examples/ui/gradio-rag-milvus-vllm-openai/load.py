import os
import re
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kubernetes import client, config
from dotenv import load_dotenv
from langchain_community.llms import VLLMOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    
# Init data
chunks_collection_name = "chunks"
summaries_collection_name = "summaries"

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

# Embedding parameters
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'nomic-ai/nomic-embed-text-v1')

# Chunk size and overlap for splitting text
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 2048))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))

# LLM parameters
MODEL_NAME = os.getenv('MODEL_NAME')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 512))
TOP_P = float(os.getenv('TOP_P', 0.95))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.01))
PRESENCE_PENALTY = float(os.getenv('PRESENCE_PENALTY', 1.03))

# Milvus parameters
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = os.getenv('MILVUS_PORT')
MILVUS_USERNAME = os.getenv('MILVUS_USERNAME')
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD')
# MILVUS_COLLECTIONS_FILE = os.getenv('MILVUS_COLLECTIONS_FILE')

print(f"MILVUS_HOST={MILVUS_HOST}, MILVUS_PORT={MILVUS_PORT}, MILVUS_USERNAME={MILVUS_USERNAME}, MILVUS_PASSWORD={MILVUS_PASSWORD}")

# Milvus collection parameters
CHUNKS_COLLECTION = os.getenv('CHUNKS_COLLECTION', 'chunks')
SUMMARIES_COLLECTION = os.getenv('SUMMARIES_COLLECTION', 'summaries')

# Summarization parameters
SUMMARY_GROUP_SIZE = int(os.getenv('SUMMARY_GROUP_SIZE', 4))

# DEFAULT_COLLECTION = os.getenv('DEFAULT_COLLECTION')
# DEFAULT_DOSSIER = 'None'
# PROMPT_FILE = os.getenv('PROMPT_FILE', 'default_prompt.txt')
# MAX_RETRIEVED_DOCS = int(os.getenv('MAX_RETRIEVED_DOCS', 4))
# SCORE_THRESHOLD = float(os.getenv('SCORE_THRESHOLD', 0.99))
    
# Load PDFs from a directory and return a list of dictionaries containing the text and metadata
def load_pdfs_from_directory(directory_path):
    docs = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                doc = {
                    'source': filename,
                    'page_count': pdf_reader.get_num_pages(),
                    'text': '',
                    'page_nums': []
                }
                index = 0
                for page_num in range(pdf_reader.get_num_pages()):
                    doc["text"] += pdf_reader.get_page(page_num).extract_text()
                    doc["page_nums"].append((index, len(doc["text"]) - 1))
                    index = len(doc["text"])
                docs.append(doc)
    return docs

def find_page_num(page_nums, index):
    for i, (start, end) in enumerate(page_nums):
        if index >= start and index <= end:
            return i
    print(f"\n\nERROR: Could not find page number for index {index}.\n\n")
    return 99999

def chunk_text(doc, chunk_size=1024, chunk_overlap=40):
    doc_chunks = []
    for i in range(0, len(doc["text"]), chunk_size):
        start_index = i
        end_index = i + chunk_size + chunk_overlap if (i + chunk_size + chunk_overlap) < len(doc["text"]) else len(doc["text"]) - 1
        doc_chunk = {
            'source': doc["source"],
            'page_count': doc["page_count"],
            'text': doc["text"][i:i + chunk_size + chunk_overlap],
            'dossier': doc["dossier"],
            'page_start': find_page_num(doc["page_nums"], start_index),
            'page_end': find_page_num(doc["page_nums"], end_index),
        }

        doc_chunks.append(doc_chunk)
    return doc_chunks

def chunk_text_lc(doc, chunk_size=1024, chunk_overlap=40):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    doc_splits = text_splitter.split_text(doc["text"])
    print(f"doc_splits={len(doc_splits)}")
    start_index = 0
    for i, split in enumerate(doc_splits):
        end_index = len(split) - 1
        doc_splits[i] = {
            'source': doc["source"],
            'page_count': doc["page_count"],
            'text': split,
            'dossier': doc["dossier"],
            'page_start': find_page_num(doc["page_nums"], start_index),
            'page_end': find_page_num(doc["page_nums"], end_index),
        }
        start_index = end_index + 1
    return doc_splits

def generate_embedding(text: str):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Add metadata to the documents which is a list of dictionaries
def add_metadata(docs):
    pattern = re.compile(r'^([0-9]+)-(.*)\.pdf$', re.IGNORECASE)
    for doc in docs:
        match = pattern.match(doc["source"])
        if match:
            doc["dossier"] = match.group(1)
        else:
            doc["dossier"] = "Unknown"
    return docs

# Initialize Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# model_kwargs = {'trust_remote_code': True}
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True) # TODO: Add trust_remote_code=True to avoid warnings
vector_size = model.config.hidden_size

# Directory path to PDFs
directory_path = '/tmp/pdfs'

# Load PDFs from the directory
docs = load_pdfs_from_directory(directory_path)
print(f"Loaded {len(docs)} PDFs from {directory_path}.")

# Print page_count and page_nums for each document
for doc in docs:
    print(f"source={doc['source']}, page_count={doc['page_count']}, page_nums={doc['page_nums']}, counted={len(doc['page_nums'])}")

# Add metadata to the documents
docs = add_metadata(docs)

# Chunk the text of each document
all_doc_chunks = []
for doc in docs:
    print(f"Chunking {doc['source']}...")
    doc_chunks = chunk_text_lc(doc, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Chunked {len(doc_chunks)} chunks from {doc['source']}.")
    print(f"sample={doc_chunks[0]}")
    all_doc_chunks.extend(doc_chunks)

print(f"Total of {len(all_doc_chunks)} chunks.")
print(f"sample={all_doc_chunks[0]}")

# Connect to Milvus
# connections.connect("default", host="localhost", port="19530", user="root", password="Milvus")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, user=MILVUS_USERNAME, password=MILVUS_PASSWORD)

# Drop collections before creating them?
drop_before_create = True

# Delete the chunks collections if it already exists and drop_before_create is True
if utility.has_collection(chunks_collection_name) and drop_before_create:
    utility.drop_collection(chunks_collection_name)

# Define the chunks collection schema
chunks_collection_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page_start", dtype=DataType.INT16),
    FieldSchema(name="page_end", dtype=DataType.INT16),
    FieldSchema(name="dossier", dtype=DataType.VARCHAR, max_length=128),
]
chunks_collection_schema = CollectionSchema(chunks_collection_fields, "Schema for storing document chunks.")

# Create a collection for chunks
chunks_collection = None
if utility.has_collection(chunks_collection_name):
    chunks_collection = Collection(chunks_collection_name)
else:
    chunks_collection = Collection(chunks_collection_name, chunks_collection_schema)

# Prepare data for insertion
vectors = [generate_embedding(doc_chunk["text"]) for doc_chunk in all_doc_chunks]

# Extract attr from the doc_chunks and store them in separate lists
sources = [doc_chunk['source'] for doc_chunk in all_doc_chunks]
texts = [doc_chunk['text'] for doc_chunk in all_doc_chunks]
page_starts = [doc_chunk['page_start'] for doc_chunk in all_doc_chunks]
page_ends = [doc_chunk['page_end'] for doc_chunk in all_doc_chunks]
dossiers = [doc_chunk['dossier'] for doc_chunk in all_doc_chunks]

# Insert data into the collection
entities = [
    vectors,
    sources,
    texts,
    page_starts,
    page_ends,
    dossiers,
]

# Print shape of the entities
print(f"Entities shape: {len(entities)} x {len(entities[0])} : {len(entities[1])} : {len(entities[2])} : {len(entities[3])} : {len(entities[4])} : {len(entities[5])}")

insert_result = chunks_collection.insert(entities)
chunks_collection.flush()

print(f"Inserted {len(insert_result.primary_keys)} records into Milvus.")

# Add an index to the collection
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"efConstruction": 200, "M": 16}  # Typical values for efConstruction and M
}
chunks_collection.create_index(field_name="vector", index_params=index_params)

# Load the collection
chunks_collection.load()

# Delete the summaries collections if it already exists and drop_before_create is True
if utility.has_collection(summaries_collection_name) and drop_before_create:
    utility.drop_collection(summaries_collection_name)

# Define the chunks collection schema
summaries_collection_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="page_start", dtype=DataType.INT16),
    FieldSchema(name="page_end", dtype=DataType.INT16),
    FieldSchema(name="dossier", dtype=DataType.VARCHAR, max_length=128),
]
summaries_collection_schema = CollectionSchema(summaries_collection_fields, "Schema for storing summaries from chunks.")

# Create a collection for summaries
summaries_collection = None
if utility.has_collection(summaries_collection_name):
    summaries_collection = Collection(summaries_collection_name)
else:
    summaries_collection = Collection(summaries_collection_name, summaries_collection_schema)

# Add an index to the collection
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"efConstruction": 200, "M": 16}  # Typical values for efConstruction and M
}
summaries_collection.create_index(field_name="vector", index_params=index_params)

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
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: {text}"
)

# Create the summarization chain
summarization_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

# Group chunks in all_doc_chunks by dossier then take chunks in groups of N and summarize them
dossier_chunks = {}
for doc_chunk in all_doc_chunks:
    dossier = doc_chunk["dossier"]
    if dossier not in dossier_chunks:
        dossier_chunks[dossier] = []
    dossier_chunks[dossier].append(doc_chunk)

# Summarize each group of N chunks
group_size = SUMMARY_GROUP_SIZE
summaries = []
for dossier, chunks in dossier_chunks.items():
    print(f"Summarizing dossier {dossier}...")
    for i in range(0, len(chunks), group_size):
        chunk_group = chunks[i:i + group_size]
        chunk_group_text = " ".join([chunk["text"] for chunk in chunk_group])
        summary_text = summarization_chain.run({"text": chunk_group_text})
        summary_vector = generate_embedding(summary_text)
        summary_source = f"{dossier}-{i}-{i + group_size - 1}"
        summary_page_start = chunk_group[0]["page_start"]
        summary_page_end = chunk_group[-1]["page_end"]
        summary_dossier = dossier

        # Insert the summary into the summaries collection
        summary_entity = [
            summary_vector,
            summary_source,
            summary_text,
            summary_page_start,
            summary_page_end,
            summary_dossier,
        ]
        insert_result = summaries_collection.insert([summary_entity])
        summaries_collection.flush()
        print(f"Inserted {summary_source} summaries into Milvus.")

# Load the collection
summaries_collection.load()