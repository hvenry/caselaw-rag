import time
import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from openai import OpenAI
from tqdm.auto import tqdm


load_dotenv()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
client = OpenAI()
embeddings = OpenAIEmbeddings()


"""
Create Pinecone index

name: The name of the index.
dimensions: The dimensions of the vectors being stored in the index (typically 768 or 1536).
metric: The distance metric to be used for similarity search.
spec: Specifies that the index should be a serverless index on AWS in us-east-1.
"""
index_name = "illinois-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Initialize the pinecone index, specify file
index = pc.Index(index_name)


# Convert chunk of text into vector embedding using OpenAI embedding
def embed_chunk(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")

    # Return embedding
    return response.data[0].embedding


# Read document from file path, split document into chunks, embed chunks
def process_document(file_path):
    # Load document with utf-8 encoding
    loader = TextLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()

    # Split document into chunks, 1000 char / chunk, overlap previous chunk by 200 chars, with starting index
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    data = text_splitter.split_documents(data)

    # Initialize results list and embed document chunks
    res = []
    print("Embedding document chunks...")

    # Iterate over each chunk (and display progress with tqdm)
    for i in tqdm(range(len(data)), desc="Processing Chunks"):
        chunk = data[i]
        # Create unique id for chunk using source and index
        id = chunk.metadata["source"] + "-" + str(i)
        # Extract text of chunk, embed into values, and store metadata
        text = chunk.page_content
        values = embed_chunk(text)
        metadata = {"text": text}
        # Add our embedded information with unique id and metadata to our result
        res.append({"id": id, "values": values, "metadata": metadata})

    # Return embedded chunks as results
    return res


# Split sequence into smaller chunks of a specified size
def chunker(seq, batch_size):
    # Yield successive chunks of sequence
    for pos in range(0, len(seq), batch_size):
        yield seq[pos : pos + batch_size]


# Function used to upload our embeddings to Pinecone
def upload_to_pinecone(data):
    async_results = []

    print("Uploading to Pinecone in batches...")

    for chunk in tqdm(
        chunker(data, batch_size=100),
        total=len(data) // 100 + 1,
        desc="Uploading Batches",
    ):
        async_results.append(index.upsert(vectors=chunk, async_req=True))

    # Wait for and retrieve responses (in case of error)
    return [async_result.result() for async_result in async_results]


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

file_path = "json_resources/sample.jsonl"


def main(file_path):
    try:
        # Process document
        logging.info("Processing document...")
        data = process_document(file_path)

        # Upload to Pinecone
        logging.info("Uploading data to Pinecone...")
        upload_to_pinecone(data)

        # Wait for all asynchronous operations to complete
        time.sleep(5)
        logging.info("Index statistics:")
        logging.info(index.describe_index_stats())

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main(file_path)
