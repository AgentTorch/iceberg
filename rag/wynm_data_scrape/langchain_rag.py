import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from typing import List, Dict, Any, Optional

# Load .env
load_dotenv()

voyage_api_key = os.getenv("VOYAGE_API_KEY")

# Loads JSON files from a directory and converts them to LangChain documents
def load_scraped_data(data_dir: str) -> List[Any]:
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(data_dir, filename)
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                text_content=False
            )
            docs.extend(loader.load())
    return docs

# Splits documents into smaller chunks for better processing
def chunk_docs(docs: List[Any]) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

# Creates a vector store using VoyageAI embeddings
def build_vectorstore(chunked_docs: List[Any]) -> Optional[Any]:
    try:
        embeddings = VoyageAIEmbeddings(model="voyage-3.5")
        vectorstore = FAISS.from_documents(chunked_docs, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

# Builds a RAG chain using Anthropic Claude and the vector store
def build_rag_chain(vectorstore: Any) -> Any:
    llm = ChatAnthropic(model="claude-3-opus-20240229")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

# Main function to retrieve results using RAG
def retrieve_results(query: str) -> str:
    data_dir = 'wynm_data_scrape/scraped_jsons'
    docs = load_scraped_data(data_dir)
    chunked_docs = chunk_docs(docs)
    vectorstore = build_vectorstore(chunked_docs)
    if vectorstore is None:
        return "Error building vectorstore"
    rag_chain = build_rag_chain(vectorstore)
    result = rag_chain.run(query)
    print(result)
    return result

# Test function using a smaller dataset
def test_voyage_api(query: str) -> str:
    data_dir = 'wynm_data_scrape/test_jsons'
    docs = load_scraped_data(data_dir)
    chunked_docs = chunk_docs(docs)
    vectorstore = build_vectorstore(chunked_docs)
    if vectorstore is None:
        return "Error building vectorstore"
    rag_chain = build_rag_chain(vectorstore)
    result = rag_chain.run(query)
    print(result)
    return result

# Example usage
if __name__ == "__main__":
    query = "What are the key points about AI in Tennessee?"
    result = retrieve_results(query)
    print(result)

# [UNUSED] Function to extract SOC code from query - not currently used in the codebase
def get_soc_code(query):
    # Extract SOC code from query
    pass

# [UNUSED] Function to get risk level from SOC code - not currently used in the codebase
def risk_from_soc_code(soc_code):
    # Get risk level from SOC code
    pass

# [UNUSED] Function to query risk information - not currently used in the codebase
def risk_query(query):
    # Query risk information
    pass

