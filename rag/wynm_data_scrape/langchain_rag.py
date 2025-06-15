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
from typing import List, Any, Optional, Literal, Dict
from tqdm.auto import tqdm
import time
import random
import pandas as pd
import glob
import json
import threading
from concurrent.futures import as_completed

# Load environment variables
load_dotenv()
voyage_api_key = os.getenv("VOYAGE_API_KEY")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'scraped_jsons')
CHUNK_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'chunk_cache.pkl')
DETAILS_CHUNK_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'details_chunk_cache.pkl')
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index')
DETAILS_FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'details_faiss_index')

# Global cache for vectorstores and documents
_vectorstore_cache = {
    'regular': None,
    'details': None
}

_document_cache = {
    'regular': None,
    'details': None
}

class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.tokens = max_requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.max_requests, self.tokens + time_passed * (self.max_requests / 60))
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# Load JSON files in parallel
def load_scraped_data_parallel(data_dir: str) -> List[Any]:
    print("\n=== Loading JSON files ===")
    docs = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(filenames)} JSON files")

    def load_file(filename):
        path = os.path.join(data_dir, filename)
        loader = JSONLoader(file_path=path, jq_schema='.', text_content=False)
        return loader.load()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_file, filenames)

    for result in results:
        docs.extend(result)
    print(f"Loaded {len(docs)} total documents")
    return docs

# Load details from JSON files
def load_details_data_parallel(data_dir: str) -> List[Any]:
    print("\n=== Loading Details from JSON files ===")
    docs = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    print(f"Found {len(filenames)} JSON files")

    def load_file(filename):
        path = os.path.join(data_dir, filename)
        # Load each section separately with metadata
        sections = {
            # Summary sections
            'summary_basic': '.summary',  # For SOC_CODE, NAME
            'technology_skills': '.summary.TechnologySkills',
            'work_context': '.summary.WorkContext',
            'professional_associations': '.summary.ProfessionalAssociations',
            'related_occupations': '.summary.RelatedOccupations',
            
            # Details sections
            'tasks': '.details.Tasks',
            'work_activities': '.details.WorkActivities',
            'skills': '.details.Skills',
            'knowledge': '.details.Knowledge',
            'abilities': '.details.Abilities',
            'interests': '.details.Interests',
            'work_values': '.details.WorkValues',
            'work_styles': '.details.WorkStyles',
            'education': '.details.Education'
        }
        
        loaded_docs = []
        for section_name, jq_schema in sections.items():
            def metadata_func(sample, additional_fields=None):
                # Get parent document to access both summary and details info
                parent_doc = additional_fields.get('parent_document', {}) if additional_fields else {}
                summary = parent_doc.get('summary', {})
                details = parent_doc.get('details', {})
                
                # Base metadata
                metadata = {
                    'section': section_name,
                    'soc_code': summary.get('soc_code', ''),
                    'job_title': summary.get('name', '')
                }
                
                # Add section-specific metadata
                if section_name == 'tasks' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'category': sample.get('Category', ''),
                        'task': sample.get('Task', '')
                    })
                elif section_name == 'work_activities' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'work_activity': sample.get('Work Activity', '')
                    })
                elif section_name == 'skills' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'skill': sample.get('Skill', '')
                    })
                elif section_name == 'knowledge' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'knowledge': sample.get('Knowledge', '')
                    })
                elif section_name == 'abilities' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'ability': sample.get('Ability', '')
                    })
                elif section_name == 'interests' and isinstance(sample, dict):
                    metadata.update({
                        'occupational_interest': sample.get('Occupational Interest', ''),
                        'interest': sample.get('Interest', '')
                    })
                elif section_name == 'work_values' and isinstance(sample, dict):
                    metadata.update({
                        'extent': sample.get('Extent', ''),
                        'work_value': sample.get('Work Value', '')
                    })
                elif section_name == 'work_styles' and isinstance(sample, dict):
                    metadata.update({
                        'importance': sample.get('Importance', ''),
                        'work_style': sample.get('Work Style', '')
                    })
                elif section_name == 'education' and isinstance(sample, dict):
                    metadata.update({
                        'level': sample.get('Level', ''),
                        'percentage': sample.get('Percentage', '')
                    })
                elif section_name == 'technology_skills' and isinstance(sample, dict):
                    metadata.update({
                        'technology_skill': sample.get('Technology Skill', '')
                    })
                elif section_name == 'work_context' and isinstance(sample, dict):
                    metadata.update({
                        'work_context': sample.get('Work Context', '')
                    })
                elif section_name == 'professional_associations' and isinstance(sample, dict):
                    metadata.update({
                        'association': sample.get('Association', '')
                    })
                elif section_name == 'related_occupations' and isinstance(sample, dict):
                    metadata.update({
                        'related_occupation': sample.get('Related Occupation', '')
                    })
                
                return metadata
            
            loader = JSONLoader(
                file_path=path,
                jq_schema=jq_schema,
                text_content=False,
                metadata_func=metadata_func
            )
            section_docs = loader.load()
            loaded_docs.extend(section_docs)
        return loaded_docs

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(load_file, filenames)

    for result in results:
        docs.extend(result)
    print(f"Loaded {len(docs)} total detail documents across all sections")
    return docs

# Smart chunking for regular content
def chunk_docs(docs: List[Any]) -> List[Any]:
    print("\n=== Chunking regular content ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks

# Smart chunking for details content
def chunk_details_docs(docs: List[Any]) -> List[Any]:
    print("\n=== Chunking details content ===")
    # Group documents by section
    section_docs = {}
    for doc in docs:
        section = doc.metadata.get('section', 'unknown')
        if section not in section_docs:
            section_docs[section] = []
        section_docs[section].append(doc)
    
    # Chunk each section separately with section-specific settings
    all_chunks = []
    chunk_settings = {
        'work_activities': {'size': 500, 'overlap': 50},  # Smaller chunks for activities
        'work_context': {'size': 500, 'overlap': 50},     # Smaller chunks for context
        'default': {'size': 1000, 'overlap': 100}         # Default settings
    }
    
    for section, section_documents in section_docs.items():
        print(f"Chunking {section} section...")
        settings = chunk_settings.get(section, chunk_settings['default'])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings['size'],
            chunk_overlap=settings['overlap'],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_documents(section_documents)
        print(f"Created {len(chunks)} chunks for {section}")
        all_chunks.extend(chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

# Use or build chunk cache
def get_or_create_chunks(docs: List[Any], is_details: bool = False) -> List[Any]:
    cache_path = DETAILS_CHUNK_CACHE_PATH if is_details else CHUNK_CACHE_PATH
    if os.path.exists(cache_path):
        print(f"\n=== Loading cached chunks from {cache_path} ===")
        with open(cache_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} cached chunks")
        return chunks
    
    print(f"\n=== Building new chunks (will be cached to {cache_path}) ===")
    chunks = chunk_details_docs(docs) if is_details else chunk_docs(docs)
    with open(cache_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"Cached {len(chunks)} chunks")
    return chunks

# Use or build vectorstore
def build_or_load_vectorstore(chunks: List[Any], is_details: bool = False) -> Optional[Any]:
    embeddings = VoyageAIEmbeddings(model="voyage-3.5")
    index_path = DETAILS_FAISS_INDEX_PATH if is_details else FAISS_INDEX_PATH
    
    if os.path.exists(index_path):
        print(f"\n=== Loading existing FAISS index from {index_path} ===")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"\n=== Building new FAISS index (will be saved to {index_path}) ===")
    try:
        print("Creating embeddings and building vectorstore...")
        # Extract texts and metadatas from chunks
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        
        # Calculate optimal batch size and workers based on 2000 RPM limit
        # Assuming each batch takes ~1 second to process
        # We want to stay under 2000 requests per minute
        # So we can do 2000/60 = ~33 requests per second
        # With 4 workers, that's ~8 requests per second per worker
        batch_size = 50  # Increased batch size since we're rate limited by time, not batch size
        max_workers = 4
        all_embeddings = []
        
        total_chunks = len(texts)
        print(f"\nProcessing {total_chunks} chunks in batches of {batch_size}")
        print("This may take several minutes...")
        
        # Create progress bar
        pbar = tqdm(
            total=total_chunks,
            desc="Creating embeddings",
            unit="chunks",
            ncols=100,
            position=0,
            leave=True
        )
        
        # Save progress to a temporary file
        progress_file = f"{index_path}_progress.pkl"
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                all_embeddings = pickle.load(f)
                pbar.update(len(all_embeddings))
                print(f"\nResuming from previous progress: {len(all_embeddings)} chunks processed")
        
        # Process in parallel using ThreadPoolExecutor
        def process_batch(batch_texts):
            max_retries = 3
            retry_delay = 5
            
            for retry in range(max_retries):
                try:
                    return embeddings.embed_documents(batch_texts)
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        print(f"\nError processing batch: {e}")
                        print(f"Retrying in {wait_time} seconds... (Attempt {retry + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"\nFailed to process batch after {max_retries} attempts")
                        raise
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create batches
            batches = [texts[i:i + batch_size] for i in range(len(all_embeddings), total_chunks, batch_size)]
            
            # Process batches in parallel with rate limiting
            for batch_embeddings in executor.map(process_batch, batches):
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch_embeddings))
                
                # Save progress after each successful batch
                with open(progress_file, 'wb') as f:
                    pickle.dump(all_embeddings, f)
                
                # No need for additional delay as the API calls themselves take enough time
                # to keep us under the 2000 RPM limit
        
        pbar.close()
        
        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print("\nBuilding FAISS index...")
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=embeddings,
            metadatas=metadatas
        )
        
        print("Saving vectorstore...")
        vectorstore.save_local(index_path)
        print("Vectorstore built and saved successfully")
        return vectorstore
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

# Build RAG chain
def build_rag_chain(vectorstore: Any) -> Any:
    print("\n=== Building RAG chain ===")
    llm = ChatAnthropic(model="claude-3-opus-20240229")
    
    # Create a retriever that can filter by section
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": None,  # Can be set to filter by section if needed
            "k": 4  # Number of chunks to retrieve
        }
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

# Main RAG query interface
def retrieve_results(query: str, use_details: bool = True) -> str:
    print(f"\n=== Processing {'details' if use_details else 'regular'} query ===")
    print(f"Query: {query}")
    
    # Check if we have a cached vectorstore
    cache_key = 'details' if use_details else 'regular'
    if _vectorstore_cache[cache_key] is None:
        print("No cached vectorstore found, building new one...")
        # Check if we have cached documents
        if _document_cache[cache_key] is None:
            print("Loading documents...")
            if use_details:
                docs = load_details_data_parallel(DATA_DIR)
            else:
                docs = load_scraped_data_parallel(DATA_DIR)
            _document_cache[cache_key] = docs
        else:
            print("Using cached documents...")
            docs = _document_cache[cache_key]
        
        chunks = get_or_create_chunks(docs, is_details=use_details)
        vectorstore = build_or_load_vectorstore(chunks, is_details=use_details)
        _vectorstore_cache[cache_key] = vectorstore
    else:
        print("Using cached vectorstore...")
        vectorstore = _vectorstore_cache[cache_key]
    
    if vectorstore is None:
        return "Vectorstore failed to build or load."
    
    rag_chain = build_rag_chain(vectorstore)
    print("\n=== Generating response ===")
    return rag_chain.run(query)

def generate_query_from_csv(csv_file: str) -> str:
    """
    Generate a query string for each row in a CSV file.
    The CSV should have columns: industry, industry_iceberg_index, occ_code, occ_title, industry_rank, avg_industry_rank
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get the type (risk/opportunity) from the filename
    filename = os.path.basename(csv_file)
    query_type = "risk" if "risk" in filename.lower() else "opportunity"
    
    # Generate queries for each row
    queries = []
    for _, row in df.iterrows():
        query = f"""Analyze the job {row['occ_code']} {row['occ_title']}. The Industry that this profession is in, {row['industry']}, has an Iceberg score ranking of {row['industry_iceberg_index']} for {query_type} in relation to AI Advancements. This ranks #{int(row['industry_rank'])} compared to the industry average #{int(row['avg_industry_rank'])}. Create a brief first paragraph justifying the Iceberg score for the industry, and relate it back to the profession. The second paragraph should USE EVIDENCE. Pull importance values for tasks, skills, knowledge, abilities, and interests, and specific education level percentages for the profession where needed to explain and reason why the score makes sense. Make comparisons, and for the last paragraph try to tie it to the broader context of the industry, and what the Iceberg score means for this state."""
        queries.append(query)
    
    return queries

def process_csv_queries(csv_file: str, use_details: bool = True) -> List[str]:
    """
    Process a CSV file and generate responses for each query in parallel with rate limiting.
    Returns a list of responses.
    """
    queries = generate_query_from_csv(csv_file)
    responses = [None] * len(queries)  # Pre-allocate list
    rate_limiter = RateLimiter(max_requests_per_minute=30)  # Adjust based on API limits
    
    def process_query(idx, query):
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            if rate_limiter.acquire():
                try:
                    response = retrieve_results(query, use_details)
                    responses[idx] = response
                    return
                except Exception as e:
                    if retry < max_retries - 1:
                        time.sleep(retry_delay * (2 ** retry))
                        continue
                    print(f"Query {idx} failed after {max_retries} retries: {e}")
                    responses[idx] = f"Error processing query: {e}"
            else:
                time.sleep(1)  # Wait if rate limit is hit
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all queries to the executor
        futures = [executor.submit(process_query, i, query) 
                  for i, query in enumerate(queries)]
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in parallel processing: {e}")
    
    return responses

def process_state_analysis(state_name: str, state_dir: str) -> Dict:
    """
    Process all CSV files for a state and structure responses into JSON format.
    Uses parallel processing with rate limiting for efficiency.
    
    Args:
        state_name: Name of the state
        state_dir: Directory containing the state's CSV files
    
    Returns:
        Dict with structure:
        {
            "state": str,
            "risk": {
                "INDUSTRY_NAME": {
                    "top_5_by_employment": List[Dict],
                    "top_5_by_economic_value": List[Dict]
                },
                ...
            },
            "opportunity": {
                "INDUSTRY_NAME": {
                    "top_5_by_employment": List[Dict],
                    "top_5_by_economic_value": List[Dict]
                },
                ...
            }
        }
    """
    # Initialize result structure
    result = {
        "state": state_name,
        "risk": {},
        "opportunity": {}
    }
    
    # Initialize vectorstore once for all workers
    print("\nInitializing vectorstore...")
    if _vectorstore_cache['details'] is None:
        if _document_cache['details'] is None:
            print("Loading documents...")
            docs = load_details_data_parallel(DATA_DIR)
            _document_cache['details'] = docs
        else:
            print("Using cached documents...")
            docs = _document_cache['details']
        
        chunks = get_or_create_chunks(docs, is_details=True)
        vectorstore = build_or_load_vectorstore(chunks, is_details=True)
        _vectorstore_cache['details'] = vectorstore
    
    # Get all CSV files for this state
    csv_pattern = os.path.join(state_dir, f"{state_name}_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Process each file
    for csv_file in csv_files:
        print(f"\nProcessing {os.path.basename(csv_file)}...")
        
        # Determine the type and ranking method from filename
        filename = os.path.basename(csv_file)
        is_risk = "risk" in filename.lower()
        is_employment = "employment" in filename.lower()
        section = "risk" if is_risk else "opportunity"
        
        # Get responses for this CSV using parallel processing
        responses = process_csv_queries(csv_file)
        
        # Read the CSV to get profession details
        df = pd.read_csv(csv_file)
        
        # Structure the responses
        for i, (_, row) in enumerate(df.iterrows()):
            industry = row['industry']
            profession_data = {
                "soc_code": row['occ_code'],
                "profession": row['occ_title'],
                "industry_iceberg_index": float(row['industry_iceberg_index']),
                "industry_rank": int(row['industry_rank']),
                "avg_industry_rank": int(row['avg_industry_rank']),
                "analysis": responses[i] if i < len(responses) else ""
            }
            
            # Initialize industry if it doesn't exist
            if industry not in result[section]:
                result[section][industry] = {
                    "top_5_by_employment": [],
                    "top_5_by_economic_value": []
                }
            
            # Add to appropriate ranking type
            ranking_type = "top_5_by_employment" if is_employment else "top_5_by_economic_value"
            result[section][industry][ranking_type].append(profession_data)
    
    return result

def save_state_analysis(state_name: str, state_dir: str, output_dir: str):
    """
    Process state data and save the analysis as JSON.
    
    Args:
        state_name: Name of the state
        state_dir: Directory containing the state's CSV files
        output_dir: Directory to save the JSON output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the state data
    analysis = process_state_analysis(state_name, state_dir)
    
    # Save to JSON file
    output_file = os.path.join(output_dir, f"{state_name}_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis saved to {output_file}")

def test_parallel_processing():
    """
    Test function to process both risk and opportunity files with parallel processing and rate limiting.
    """
    # Test files
    test_files = [
        "../state_iceberg_data/Tennessee_4.82_risk_top_5_by_employment.csv",
        "../state_iceberg_data/Tennessee_4.82_opportunity_top_5_by_employment.csv"
    ]
    
    # Initialize vectorstore once for all workers
    print("\nInitializing vectorstore...")
    if _vectorstore_cache['details'] is None:
        if _document_cache['details'] is None:
            print("Loading documents...")
            docs = load_details_data_parallel(DATA_DIR)
            _document_cache['details'] = docs
        else:
            print("Using cached documents...")
            docs = _document_cache['details']
        
        chunks = get_or_create_chunks(docs, is_details=True)
        vectorstore = build_or_load_vectorstore(chunks, is_details=True)
        _vectorstore_cache['details'] = vectorstore
    
    # Create a test result structure with industry nesting
    test_result = {
        "test_files": [os.path.basename(f) for f in test_files],
        "risk": {},
        "opportunity": {}
    }
    
    # Process each file
    for test_file in test_files:
        print(f"\nProcessing {os.path.basename(test_file)}...")
        responses = process_csv_queries(test_file)
        
        # Determine if this is a risk or opportunity file
        is_risk = "risk" in test_file.lower()
        section = "risk" if is_risk else "opportunity"
        
        # Read the CSV to get profession details
        df = pd.read_csv(test_file)
        
        # Structure the responses by industry
        for i, (_, row) in enumerate(df.iterrows()):
            industry = row['industry']
            profession_data = {
                "soc_code": row['occ_code'],
                "profession": row['occ_title'],
                "industry_iceberg_index": float(row['industry_iceberg_index']),
                "industry_rank": int(row['industry_rank']),
                "avg_industry_rank": int(row['avg_industry_rank']),
                "analysis": responses[i] if i < len(responses) else ""
            }
            
            # Initialize industry if it doesn't exist
            if industry not in test_result[section]:
                test_result[section][industry] = {
                    "top_5_by_employment": [],
                    "top_5_by_economic_value": []
                }
            
            # Add to employment list since these are employment files
            test_result[section][industry]["top_5_by_employment"].append(profession_data)
    
    # Save test result
    output_dir = "../test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_parallel_output.json")
    
    with open(output_file, 'w') as f:
        json.dump(test_result, f, indent=2)
    
    print(f"\nTest output saved to {output_file}")
    return test_result

# Test version
if __name__ == "__main__":
    # Process Tennessee data
    state_name = "Tennessee"
    state_dir = os.path.join(os.path.dirname(__file__), "..", "state_iceberg_data")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "state_analyses")
    
    print(f"\nProcessing {state_name} data...")
    print(f"Looking for CSV files in: {state_dir}")
    print(f"Will save output to: {output_dir}")
    
    save_state_analysis(state_name, state_dir, output_dir)
