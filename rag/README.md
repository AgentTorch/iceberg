# RAG Pipeline for Job Analysis

This directory contains the RAG (Retrieval-Augmented Generation) pipeline for analyzing job data and generating insights based on Iceberg scores.

## Essential Files

- `langchain_rag.py`: Main RAG implementation for processing job data and generating analyses
- `occupation_utils.py`: Utility functions for processing occupation data
- `scrape_v2.py`: Script for scraping job data (if needed)

## Setup

1. Install required packages:
```bash
pip install python-dotenv langchain-community langchain-anthropic langchain-voyageai faiss-cpu pandas tqdm
```

2. Create a `.env` file with your API keys:
```
VOYAGE_API_KEY=your_voyage_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

3. The first time you run the script, it will:
   - Load and process the job data
   - Create embeddings
   - Build the FAISS index
   - Cache the results for future use

## Usage

To process a state's data:
```python
from langchain_rag import save_state_analysis

state_name = "Tennessee"  # or any other state
state_dir = "../state_iceberg_data"
output_dir = "../state_analyses"

save_state_analysis(state_name, state_dir, output_dir)
```

## Output

The script generates a JSON file for each state with the following structure:
```json
{
    "state": "StateName",
    "risk": {
        "top_5_by_employment": [...],
        "top_5_by_economic_value": [...]
    },
    "opportunity": {
        "top_5_by_employment": [...],
        "top_5_by_economic_value": [...]
    }
}
```

## Note

The following files are gitignored as they are generated during runtime:
- Cache files (*.pkl)
- FAISS indices
- Scraped data
- Python cache files
- Environment files
- IDE files
- Log files 