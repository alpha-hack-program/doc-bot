import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

def load_files(download_path: str):
    # Process the downloaded PDFs
    pdfs_to_urls = {}
    for file_name in os.listdir(download_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(download_path, file_name)
            print(f"Processing file: {file_path}")
            # Remove extension from file name
            file_name = file_name.replace('.pdf', '')
            # Add the url of the PDF in the S3 bucket to pdfs_to_urls dictionary
            pdfs_to_urls[file_name] = file_path
        
    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(download_path)
    pdf_docs = pdf_loader.load()

    # Inject metadata
    from pathlib import Path

    for doc in pdf_docs:
        doc.metadata["source"] = pdfs_to_urls[Path(doc.metadata["source"]).stem]

    # Split documents into chunks with some overlap
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(pdf_docs)

    # Create the index and ingest the documents
    model_kwargs = {'trust_remote_code': True, 'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs=model_kwargs,
        show_progress=True
    )

    milvus_host = os.getenv('MILVUS_HOST')
    milvus_port = os.getenv('MILVUS_PORT')
    milvus_username = os.getenv('MILVUS_USERNAME')
    milvus_password = os.getenv('MILVUS_PASSWORD')
    milvus_collection = os.getenv('MILVUS_COLLECTION_NAME')

    db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_username, "password": milvus_password},
        collection_name=milvus_collection,
        metadata_field="metadata",
        text_field="page_content",
        auto_id=True,
        drop_old=True
    )
    
    db.add_documents(all_splits)

    print("done")
