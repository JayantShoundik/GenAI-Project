# FINAL SCRIPT FOR DAY 3

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. Load the document ---
print("Loading document...")
# You need to have a 'mydocument.txt' file in the same folder
loader = TextLoader('mydocument.txt')
documents = loader.load()

# --- 2. Split the document into chunks ---
print("Splitting document into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Document split into {len(chunks)} chunks.")

# --- 3. Create embeddings and store in ChromaDB ---
print("Creating embeddings...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

print("Persisting vectors to vector store...")
# This creates a 'chroma_db' folder to store the vectorized data
db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

print("--- Day 3 Complete! ---")