import os
import random
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
Pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize Pinecone Vector Store
def init_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # we can change this if we wish
    index_name = "test-index"
    vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    return vector_store

# Initialize memory for conversation history
def init_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

# Generate the structured prompt
def generate_prompt(query, context, history):
    prompt = f"""
    You are a legal assistant. Answer caselaw questions based on the following:

    **Context**:
    {context}

    **Conversation History**:
    {history}

    **User Query**:
    {query}

    Only provide answers based on the information provided. Only output your response. If the user indicates they are done, please end the chat using the 'end_chat' function.
    """
    return prompt

# Get the conversation history
def get_history(memory):
    messages = memory.chat_memory.messages
    history = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history += f"LawBot: {msg.content}\n"
    return history.strip()

# Retrieve context from Pinecone using the retriever
def get_context(query, retriever):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    return context

# End the chat gracefully
def end_chat():
    print("LawBot: Okay! Let me know if you need anything else!")
    exit()

# Retrieve response from OpenAI and handle function calls
def get_openai_response(full_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": full_prompt}
        ],
        functions=[
            {
                "name": "end_chat",
                "description": "Ends the chat session when the user wants to quit."
            }
        ],
        function_call="auto",
        temperature=1
    )
    # Check for function call
    if response.choices[0].finish_reason == "function_call":
        if response.choices[0].message.function_call.name == "end_chat":
            end_chat()

    return response.choices[0]

# Chat function using retriever and function calling
def chat_with_rag(query, retriever, memory):

    # Step 1: Get conversation history
    history = get_history(memory)

    # Step 2: Retrieve relevant context from Pinecone
    context = get_context(query, retriever)

    # Step 3: Generate the prompt with structured sections
    full_prompt = generate_prompt(query, context, history)

    # Step 4: Get the response from OpenAI
    response = get_openai_response(full_prompt)

    return response, context

# Main function to drive the chatbot
def main():
    # Initialize components
    vector_store = init_vector_store()
    memory = init_memory()

    # Initialize retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})

    # Welcome message
    print("LawBot: Hi! How can I help you with your legal questions?")

    while True:
        query = input("User: ")

        output, _ = chat_with_rag(query, retriever, memory)
        print(f"LawBot: {output.message.content}")

        # Update memory with the user's query and assistant's response
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(output.message.content)

if __name__ == "__main__":
    main()
