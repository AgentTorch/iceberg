import json
import os
import pandas as pd
from typing import List, Dict
import glob

def get_industry_iceberg_index(data: Dict, industry: str) -> float:
    """Get the iceberg index for a specific industry."""
    industry_breakdown = data.get('state_iceberg_index', {}).get('industry_breakdown', {})
    return industry_breakdown.get(industry, {}).get('industry_iceberg_index', 0)

def get_top_5_jobs(data: Dict, section: str) -> List[Dict]:
    """Extract top 5 jobs from either risk or opportunity section."""
    jobs = []
    if section in data and 'industry_wise_top_5_jobs' in data[section]:
        for industry, industry_data in data[section]['industry_wise_top_5_jobs'].items():
            industry_iceberg = get_industry_iceberg_index(data, industry)
            # Get jobs from both employment and economic value rankings
            for ranking_type in ['top_5_by_employment', 'top_5_by_economic_value']:
                if ranking_type in industry_data:
                    for job in industry_data[ranking_type]:
                        jobs.append({
                            'industry': industry,
                            'industry_iceberg_index': industry_iceberg,
                            'occ_code': job.get('OCC_CODE', ''),
                            'occ_title': job.get('OCC_TITLE', ''),
                            'ranking_type': ranking_type
                        })
    return jobs

def add_industry_rankings_and_split(csv_dir: str):
    """
    Add industry rankings to all risk and opportunity CSV files and split them by ranking type.
    Rankings are calculated per state, with higher iceberg indices getting higher ranks.
    """
    # Get all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    # First, calculate average industry ranks across all states
    all_industry_indices = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for industry, group in df.groupby('industry'):
            if industry not in all_industry_indices:
                all_industry_indices[industry] = []
            all_industry_indices[industry].append(group['industry_iceberg_index'].iloc[0])
    
    # Calculate average index and rank for each industry
    industry_avg_ranks = {}
    for industry, indices in all_industry_indices.items():
        avg_index = sum(indices) / len(indices)
        industry_avg_ranks[industry] = avg_index
    
    # Calculate ranks based on average index
    sorted_industries = sorted(industry_avg_ranks.items(), key=lambda x: x[1], reverse=True)
    for rank, (industry, _) in enumerate(sorted_industries, 1):
        industry_avg_ranks[industry] = rank
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing {os.path.basename(csv_file)}...")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get unique industries and their iceberg indices
        industry_indices = df[['industry', 'industry_iceberg_index']].drop_duplicates()
        
        # Calculate ranks (higher index = higher rank)
        industry_indices['industry_rank'] = industry_indices['industry_iceberg_index'].rank(ascending=False, method='min')
        
        # Create a mapping of industry to rank
        rank_mapping = dict(zip(industry_indices['industry'], industry_indices['industry_rank']))
        
        # Add rank column to the original dataframe
        df['industry_rank'] = df['industry'].map(rank_mapping)
        
        # Add average industry rank
        df['avg_industry_rank'] = df['industry'].map(industry_avg_ranks)
        
        # Split by ranking type
        for ranking_type in ['top_5_by_employment', 'top_5_by_economic_value']:
            # Filter rows for this ranking type
            filtered_df = df[df['ranking_type'] == ranking_type].copy()
            
            # Remove the ranking_type column since it's now in the filename
            filtered_df = filtered_df.drop('ranking_type', axis=1)
            
            # Create new filename
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            new_filename = f"{base_name}_{ranking_type}.csv"
            new_filepath = os.path.join(csv_dir, new_filename)
            
            # Save the filtered dataframe
            filtered_df.to_csv(new_filepath, index=False)
            print(f"Created {new_filename}")
        
        # Remove the original file
        os.remove(csv_file)
        print(f"Removed original file {os.path.basename(csv_file)}")

def process_state_file(file_path: str) -> Dict:
    """Process a single state file and return relevant data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    state_name = data.get('state', '')
    state_iceberg_index = data.get('state_iceberg_index', {}).get('state_iceberg_index', 0)
    
    # Get jobs from both risk and opportunity sections
    risk_jobs = get_top_5_jobs(data, 'risk')
    opportunity_jobs = get_top_5_jobs(data, 'opportunity')
    
    return {
        'state': state_name,
        'state_iceberg_index': state_iceberg_index,
        'risk_jobs': risk_jobs,
        'opportunity_jobs': opportunity_jobs
    }

def analyze_industry_rankings(csv_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Analyze industry rankings across all states and calculate relative rankings.
    Returns a dictionary with industry rankings and statistics.
    """
    # Dictionary to store industry data
    industry_data = {}
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, '*_opportunity_*.csv'))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        state_name = os.path.basename(csv_file).split('_')[0]
        
        # Group by industry and get unique iceberg indices
        for industry, group in df.groupby('industry'):
            if industry not in industry_data:
                industry_data[industry] = {
                    'indices': [],
                    'states': [],
                    'avg_index': 0,
                    'rank': 0
                }
            
            # Get unique iceberg index for this industry in this state
            unique_indices = group['industry_iceberg_index'].unique()
            if len(unique_indices) > 0:
                industry_data[industry]['indices'].append(unique_indices[0])
                industry_data[industry]['states'].append(state_name)
    
    # Calculate average index and rank for each industry
    for industry in industry_data:
        if industry_data[industry]['indices']:
            industry_data[industry]['avg_index'] = sum(industry_data[industry]['indices']) / len(industry_data[industry]['indices'])
    
    # Calculate ranks based on average index
    sorted_industries = sorted(industry_data.items(), key=lambda x: x[1]['avg_index'], reverse=True)
    for rank, (industry, _) in enumerate(sorted_industries, 1):
        industry_data[industry]['rank'] = rank
    
    return industry_data

def main():
    # Get the absolute path of the workspace root
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Directory containing the CSV files
    output_dir = os.path.join(workspace_root, 'rag', 'state_iceberg_data')
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist!")
        return
    
    # Add industry rankings and split CSV files
    print("\nAdding industry rankings and splitting CSV files...")
    add_industry_rankings_and_split(output_dir)
    
    # Analyze industry rankings across states
    print("\nAnalyzing industry rankings across states...")
    industry_rankings = analyze_industry_rankings(output_dir)
    
    # Print summary of industry rankings
    print("\n=== Industry Rankings Summary ===")
    for industry, data in sorted(industry_rankings.items(), key=lambda x: x[1]['rank']):
        print(f"{industry}: Rank {data['rank']} (Avg Index: {data['avg_index']:.2f})")

if __name__ == "__main__":
    main()
