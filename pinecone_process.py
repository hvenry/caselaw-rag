from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()
client = OpenAI()
embeddings = OpenAIEmbeddings()
file_path = "json_resources/sample.jsonl"  # change this depending on the document we want to upload

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = "illinois-index"  # change this to align with the index we want

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # this is dependent on the embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
index = pc.Index(index_name)

def embed_chunk(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_document(file_path):
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    data = text_splitter.split_documents(data)
    res = []
    
    print("Embedding document chunks...")
    for i in tqdm(range(len(data)), desc="Processing Chunks"):
        chunk = data[i]
        id = chunk.metadata["source"] + "-" + str(i)
        text = chunk.page_content
        values = embed_chunk(text)
        metadata = {"text": text}
        res.append({"id": id, "values": values, "metadata": metadata})
    
    return res

def chunker(seq, batch_size):
    return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

def upload_to_pinecone(data):
    async_results = []
    
    print("Uploading to Pinecone in batches...")
    for chunk in tqdm(chunker(data, batch_size=100), total=len(data) // 100 + 1, desc="Uploading Batches"):
        async_results.append(index.upsert(vectors=chunk, async_req=True))
    
    # Wait for and retrieve responses (in case of error)
    return [async_result.result() for async_result in async_results]

# For Single Files
print("Processing document...")
data = process_document(file_path)
print("Uploading data to Pinecone...")
upload_to_pinecone(data)

time.sleep(5)
print(index.describe_index_stats())
