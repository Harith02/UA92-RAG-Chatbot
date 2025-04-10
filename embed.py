import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "ua92_embeddings"
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("Please set your OpenAI API key in the environment.")

def load_docs_from_file(filename="crawled_docs.json"):
    with open(filename, "r", encoding="utf-8") as f:
        docs_serializable = json.load(f)
    # Convert back to LangChain Document objects
    docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in docs_serializable]
    return docs

def main():
    # Load crawled documents
    print("📂 Loading crawled documents...")
    all_docs = load_docs_from_file()
    print(f"Loaded {len(all_docs)} documents.")

    # Split documents into chunks
    print("🔪 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Create embeddings
    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectorstore.persist()
    print(f"💾 Embeddings saved to: {PERSIST_DIR}")

if __name__ == "__main__":
    main()