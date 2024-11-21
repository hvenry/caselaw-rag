# Case Law RAG

This is a project made for CISC 452 to explore the capabilities of RAG ([Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)) in the context of [common law](https://en.wikipedia.org/wiki/Case_law#:~:text=Case%20law%2C%20also%20used%20interchangeably,constitutions%2C%20statutes%2C%20or%20regulations).

---

## Features
- **OpenAI GPT Integration**: Provides intelligent responses based on legal context and user query.
- **Pinecone Indexing**: Retrieves relevant legal documents for accurate responses.
- **Conversation History**: Maintains chat history to provide contextual replies.
- **Unstructured Data Processing**: Converts raw legal documents into structured embeddings for retrieval.

---

## Requirements
- **Python Version**: 3.10 or higher
- **Dependencies**: Listed in `requirements.txt`

Install all required packages:
```bash
pip install -r requirements.txt
```

Environment variables:
- `OPENAI_API_KEY`: OpenAI API key.
- `PINECONE_API_KEY`: Pinecone API key.
- `UNSTRUCTURED_API_KEY`: Unstructured API key (for document partitioning).
- `UNSTRUCTURED_API_URL`: Unstructured API endpoint.

---

## File Descriptions

### 1. `caselaw_RAG.py`
The main chatbot script:
- Initializes Pinecone and OpenAI clients.
- Manages conversation history using LangChain.
- Retrieves context from indexed documents.
- Generates responses to user queries.

### 2. `pinecone_process.py`
Handles embedding and indexing of documents:
- Uses OpenAI embeddings to convert text into vector representations.
- Uploads embeddings to Pinecone in batch mode.
- Includes utilities for processing single documents or directories.

### 3. `unstructured.py`
Processes raw legal documents:
- Partitions documents into manageable chunks using the `unstructured_ingest` library.
- Configures high-resolution partitioning with API support.
- Converts documents into JSON format for indexing.

### 4. `requirements.txt`
A list of dependencies required to run the project.

---

## Setup

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a `.env` file in the project root:
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

---

## Usage

### Running the Chatbot
Start the chatbot:
```bash
python caselaw_RAG.py
```

Interact with the chatbot:
```
LawBot: Hi! How can I help you with your legal questions?
User: What is the ruling in Brown v. Board of Education?
LawBot: [Response based on retrieved context]
```

---

## Customization

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

---

## Troubleshooting

### Missing API Keys
Ensure that the `.env` file is correctly configured with the necessary API keys.

### Pinecone Index Not Found
If the specified Pinecone index does not exist, create it manually or ensure `pinecone_process.py` is run correctly.

### Unstructured API Errors
Verify that the `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_API_URL` are correctly set in the `.env` file.


---

## Acknowledgments
This project uses:
- [OpenAI](https://openai.com) for GPT-based responses and embeddings.
- [Pinecone](https://www.pinecone.io/) for vector storage and retrieval.
- [LangChain](https://langchain.com/) for conversational memory and RAG pipelines.
- [Unstructured](https://github.com/Unstructured-IO/unstructured) for document processing.

---