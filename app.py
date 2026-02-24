import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

# Initialize Embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Vector Store
index_name = "medical-chatbot"
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize Retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True) # Note: stable version is 1.5-flash

# Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the Chains
# 1. This chain handles how the retrieved documents are fed to the LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 2. This chain combines the retriever and the QA chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # Invoke the RAG chain
    response = rag_chain.invoke({"input": msg})
    
    print("Response: ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)