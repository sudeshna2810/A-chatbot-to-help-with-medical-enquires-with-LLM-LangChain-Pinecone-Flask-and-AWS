# reset_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")  # Optional: set if needed

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# List existing indexes
existing_indexes = pc.list_indexes().names()
print("Existing indexes:", existing_indexes)

# Delete the 'medical-chatbot' index if it exists
if 'medical-chatbot' in existing_indexes:
    pc.delete_index('medical-chatbot')
    print("Deleted existing index 'medical-chatbot'.")
else:
    print("No index named 'medical-chatbot' found. Nothing to delete.")