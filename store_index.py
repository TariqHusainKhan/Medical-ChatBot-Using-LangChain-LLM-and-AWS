from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

project_root = r"C:\VIZUARA-ML-AI\projects\medical_chatBot"

print("Loading PDFs...")
extracted_data = load_pdf_files(os.path.join(project_root, "data"))
print(f"Loaded {len(extracted_data)} pages")

minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(minimal_docs)
print(f"Created {len(text_chunks)} chunks")

embeddings = download_embeddings()
print("Embedding model ready")

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

print("Upserting to Pinecone (this takes a while)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,   # you assigned 'embeddings', not 'embedding'
    index_name=index_name
)

print(f"Upserted {len(text_chunks)} chunks to Pinecone index '{index_name}'")
print("Done!")