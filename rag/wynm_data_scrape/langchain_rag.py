# rag_pipeline.py
import os
import pickle
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from typing import List, Any, Optional

# Load environment variables
load_dotenv()
voyage_api_key = os.getenv("VOYAGE_API_KEY")

# Paths
DATA_DIR = 'wynm_data_scrape/scraped_jsons'
CHUNK_CACHE_PATH = 'chunk_cache.pkl'
FAISS_INDEX_PATH = 'faiss_index'

# Load JSON files in parallel
def load_scraped_data_parallel(data_dir: str) -> List[Any]:
    docs = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    def load_file(filename):
        path = os.path.join(data_dir, filename)
        loader = JSONLoader(file_path=path, jq_schema='.', text_content=False)
        return loader.load()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_file, filenames)

    for result in results:
        docs.extend(result)
    return docs

# Smart chunking
def chunk_docs(docs: List[Any]) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

# Use or build chunk cache
def get_or_create_chunks(docs: List[Any]) -> List[Any]:
    if os.path.exists(CHUNK_CACHE_PATH):
        with open(CHUNK_CACHE_PATH, 'rb') as f:
            return pickle.load(f)
    chunks = chunk_docs(docs)
    with open(CHUNK_CACHE_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    return chunks

# Use or build vectorstore
def build_or_load_vectorstore(chunks: List[Any]) -> Optional[Any]:
    embeddings = VoyageAIEmbeddings(model="voyage-3.5")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

# Build RAG chain
def build_rag_chain(vectorstore: Any) -> Any:
    llm = ChatAnthropic(model="claude-3-opus-20240229")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

# Main RAG query interface
def retrieve_results(query: str) -> str:
    docs = load_scraped_data_parallel(DATA_DIR)
    chunks = get_or_create_chunks(docs)
    vectorstore = build_or_load_vectorstore(chunks)
    if vectorstore is None:
        return "Vectorstore failed to build or load."
    rag_chain = build_rag_chain(vectorstore)
    return rag_chain.run(query)

# Test version
if __name__ == "__main__":
    query = "What are the key points about AI in Tennessee?"
    response = retrieve_results(query)
    print("\n=== Answer ===\n")
    print(response)
