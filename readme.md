# Case Law RAG

This is a project made for CISC 452 to explore the capabilities of RAG ([Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)) in the context of [common law](https://en.wikipedia.org/wiki/Case_law).

## Features

- **OpenAI GPT Integration**: Provides intelligent responses based on legal context and user query.
- **Pinecone Indexing**: Retrieves relevant legal documents for accurate responses.
- **Conversation History**: Maintains chat history to provide contextual replies.
- **Unstructured Data Processing**: Converts raw legal documents into structured embeddings for retrieval.

### This project uses:

- [OpenAI](https://openai.com) for GPT-based responses and embeddings.
- [Pinecone](https://www.pinecone.io/) for vector storage and retrieval.
- [LangChain](https://langchain.com/) for conversational memory and RAG pipelines.
- [Unstructured](https://github.com/Unstructured-IO/unstructured) for document processing.

## Case Law RAG Setup

### 1. Clone and navigate inside of the repository

```bash
git clone https://github.com/hvenry/caselaw-rag.git
```
```bash
cd caselaw-rag
```

### 2. Setup Local Environment with Conda

Create new conda environment:

```bash
conda create --name caselaw_rag python=3.10
```

Activate the conda environment:

```bash
conda activate caselaw_rag
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root with the appropriate keys:

```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
UNSTRUCTURED_API_KEY=your-unstructured-api-key
UNSTRUCTURED_API_URL=your-unstructured-api-url
```

### Step 4: Process Documents for Indexing

Use `unstructured.py` to partition raw documents:

```bash
python unstructured.py
```

Upload the processed documents to Pinecone:

```bash
python pinecone_process.py
```

## Case Law RAG Usage

In the root of the project directory with your virtual environment active installed with appropriate dependencies use:

```bash
python caselaw_RAG.py
```

You should now be prompted with the following in your CLI:

```
LawBot: Hi! How can I help you with your legal questions?
User: What is the ruling in Brown v. Board of Education?
LawBot: [Response based on retrieved context]
```

## Dependencies:

- `openai`: Python client library for accessing OpenAI's API.
- `lanchain`: Framework for building applications with LLMs.
- `langchain_openai`: Module within LangChain specifically designed for integrating with OpenAI's API.
- `langchain_pinecone`: Module within LangChain for integrating with Pinecone.
- `pinecone[grpc]`: Extension of Pinecone client that includes gRPC support for efficient communication with Pinecone servers.
- `python-dotenv`: Library for loading variables from `.env` file into the environment.
- `tqdm`: Library for creating progress bars in Python applications.

## Environment variables:

- `OPENAI_API_KEY`: OpenAI API key.
- `PINECONE_API_KEY`: Pinecone API key.
- `UNSTRUCTURED_API_KEY`: Unstructured API key (for document partitioning).
- `UNSTRUCTURED_API_URL`: Unstructured API endpoint.

## File Descriptions

### The main chatbot script: `caselaw_RAG.py`

- Initializes Pinecone and OpenAI clients.
- Manages conversation history using LangChain.
- Retrieves context from indexed documents.
- Generates responses to user queries.

### Embedding and indexing of documents: `pinecone_process.py`

- Uses OpenAI embeddings to convert text into vector representations.
- Uploads embeddings to Pinecone in batch mode.
- Includes utilities for processing single documents or directories.

### Processes raw legal documents: `unstructured.py`

- Partitions documents into manageable chunks using the `unstructured_ingest` library.
- Configures high-resolution partitioning with API support.
- Converts documents into JSON format for indexing.

## RAG Customization

### Changing Embedding Model

To use a different OpenAI embedding model, modify the `model` parameter in `caselaw_RAG.py` and `pinecone_process.py`:

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Adjusting Text Chunking

In `pinecone_process.py`, modify the chunk size and overlap:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

## Troubleshooting

### Missing API Keys

- Make sure that the `.env` file is correctly configured with the necessary API keys.

### Pinecone Index Not Found

- If the specified Pinecone index does not exist, create it manually or ensure `pinecone_process.py` is run correctly.

### Unstructured API Errors

- Verify that the `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_API_URL` are correctly set in the `.env` file.
