import time
import os
import logging
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
index_name = "ny-case-law-index"
file_path = "data/caselaw_dataset.json"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Initialize the pinecone index
index = pc.Index(index_name)


# Convert chunk of text into vector embedding using OpenAI embedding
def embed_chunk(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    # Return embedding
    return response.data[0].embedding


# Process a single case to extract key information and embed it
def process_case(case_data):
    case_id = case_data.get("id", "unknown_id")
    case_name = case_data.get("name", "unknown_case")
    legal_issues = case_data.get("casebody", {}).get("opinions", [{}])[0].get("text", "")
    summary = f"Case Name: {case_name}\nLegal Issues: {legal_issues}"

    # Split the summary into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_text(summary)

    # Embed each chunk
    embeddings = []
    for idx, chunk in enumerate(chunks):
        values = embed_chunk(chunk)
        metadata = {"case_id": case_id, "case_name": case_name, "text": chunk}
        embeddings.append({"id": f"{case_id}-{idx}", "values": values, "metadata": metadata})
    return embeddings


# Process a JSON file containing multiple cases
def process_cases(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    all_embeddings = []
    print("Processing cases...")
    for case in tqdm(cases, desc="Embedding Cases"):
        embeddings = process_case(case)
        all_embeddings.extend(embeddings)
    return all_embeddings


# Split sequence into smaller chunks of a specified size
def chunker(seq, batch_size):
    # Yield successive chunks of sequence
    for pos in range(0, len(seq), batch_size):
        yield seq[pos:pos + batch_size]


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



def main(file_path):
    try:
        # Process cases
        logging.info("Processing cases...")
        data = process_cases(file_path)

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
