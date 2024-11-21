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
file_path = "resources/wondervector.md" # change this depending on the document we want to upload

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

index_name = "test-index" # change this to algin with the index we want

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension= 3072, # this is dependent on the embedding model
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
    model="text-embedding-3-large"
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
    for i in range(len(data)):
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
    async_results = [
    index.upsert(vectors=chunk, async_req=True)
    for chunk in chunker(data, batch_size=100)
    ]
    # Wait for and retrieve responses (in case of error)
    return [async_result.result() for async_result in async_results]


# For Directories
# for filename in os.listdir("json_resources"):
#     if filename.endswith(".json"):
#         data = process_document(f"json_resources/{filename}")
#         upload_to_pinecone(data)

# For Single Files
data = process_document(file_path)
upload_to_pinecone(data)
time.sleep(5)

print(index.describe_index_stats())