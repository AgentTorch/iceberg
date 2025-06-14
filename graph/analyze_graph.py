import os
import json
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
from build_graph import build_graph, build_paths_for_threshold

def analyze_high_risk_professions(threshold=60, num_branches=3):
    """
    Analyze the graph to find the most frequently occurring professions in high-risk paths.
    
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
    
    Returns:
        List of tuples (soc_code, count, risk_score, label) sorted by count in descending order
    """
    # Build the graph and get paths
    G = build_graph()
    paths = build_paths_for_threshold(G, threshold, num_branches)
    
    # Collect all nodes from paths and branches
    all_nodes = []
    for path_info in paths:
        # Add nodes from main path
        all_nodes.extend(path_info['path'])
        # Add nodes from branches
        for branch in path_info['longest_branches']:
            all_nodes.extend(branch)
    
    # Count occurrences of each node
    node_counts = Counter(all_nodes)
    
    # Create list of tuples with node info
    node_info = []
    for soc_code, count in node_counts.items():
        node_data = G.nodes[soc_code]
        node_info.append((
            soc_code,
            count,
            node_data['automation_risk_score'],
            node_data['label']
        ))
    
    # Sort by count in descending order
    node_info.sort(key=lambda x: x[1], reverse=True)
    
    return node_info

def analyze_path_medians(threshold=60, num_branches=3):
    """
    Analyze the median risk scores along each path and identify jobs with highest median risks.
    
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
    
    Returns:
        List of tuples (soc_code, median_risk, path_length, label) sorted by median risk in descending order
    """
    # Build the graph and get paths
    G = build_graph()
    paths = build_paths_for_threshold(G, threshold, num_branches)
    
    # Analyze each path
    path_medians = []
    for path_info in paths:
        # Get all nodes in the path (main path + branches)
        path_nodes = set(path_info['path'])
        for branch in path_info['longest_branches']:
            path_nodes.update(branch)
        
        # Calculate risk scores for all nodes in the path
        risk_scores = [G.nodes[node]['automation_risk_score'] for node in path_nodes]
        median_risk = np.median(risk_scores)
        
        # Get the starting node info
        start_node = path_info['start']
        start_data = G.nodes[start_node]
        
        path_medians.append((
            start_node,  # SOC code
            median_risk,  # Median risk score
            len(path_nodes),  # Number of nodes in path
            start_data['label'],  # Job title
            risk_scores  # List of all risk scores in path
        ))
    
    # Sort by median risk in descending order
    path_medians.sort(key=lambda x: x[1], reverse=True)
    
    return path_medians

def calculate_mean_risk_scores(G, paths):
    """
    Calculate mean risk scores for each path.
    
    Args:
        G: NetworkX graph
        paths: List of paths, where each path is a list of risk scores
        
    Returns:
        List of tuples (soc_code, mean_risk, path_length, job_title, path)
        Only includes the highest mean risk path for each unique starting job
    """
    # Dictionary to store the best path for each starting job
    best_paths = {}
    
    # Get all nodes with their risk scores for lookup
    node_risk_map = {node: data.get('automation_risk_score', 0) 
                    for node, data in G.nodes(data=True)}
    
    for path in paths:
        if not path:  # Skip empty paths
            continue
            
        # Calculate mean risk score for the path
        risk_scores = [score for score in path if isinstance(score, (int, float))]
        if not risk_scores:  # Skip if no valid risk scores
            continue
            
        mean_risk = sum(risk_scores) / len(risk_scores)
        
        # Find the node that matches the first risk score in the path
        first_score = path[0]
        matching_nodes = [node for node, score in node_risk_map.items() 
                         if abs(score - first_score) < 0.001]  # Use small epsilon for float comparison
        
        if matching_nodes:
            soc_code = matching_nodes[0]
            job_title = G.nodes[soc_code].get('label', 'Unknown')
            
            # Only keep this path if it's the first one for this job or has a higher mean risk
            if soc_code not in best_paths or mean_risk > best_paths[soc_code][1]:
                best_paths[soc_code] = (soc_code, mean_risk, len(path), job_title, path)
    
    # Convert dictionary values to list and sort by mean risk score in descending order
    path_means = list(best_paths.values())
    return sorted(path_means, key=lambda x: x[1], reverse=True)

def save_analysis_results(node_info, output_file='high_risk_analysis.csv'):
    """
    Save the analysis results to a CSV file.
    
    Args:
        node_info: List of tuples (soc_code, count, risk_score, label)
        output_file: Path to output CSV file
    """
    df = pd.DataFrame(node_info, columns=['SOC_Code', 'Occurrence_Count', 'Risk_Score', 'Job_Title'])
    df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

def save_median_analysis(path_medians, output_file='path_median_analysis.csv'):
    """
    Save the median risk analysis results to a CSV file.
    
    Args:
        path_medians: List of tuples (soc_code, median_risk, path_length, label, risk_scores)
        output_file: Path to output CSV file
    """
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'SOC_Code': soc_code,
            'Median_Risk': median_risk,
            'Path_Length': path_length,
            'Job_Title': label,
            'Risk_Scores': ','.join(map(str, risk_scores))
        }
        for soc_code, median_risk, path_length, label, risk_scores in path_medians
    ])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Median analysis results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    G = build_graph()
    node_info = analyze_high_risk_professions(threshold=60, num_branches=3)
    path_medians = analyze_path_medians(threshold=60, num_branches=3)
    mean_scores = calculate_mean_risk_scores(G, [path for _, _, _, _, path in path_medians])
    
    print("Top 10 Most Frequently Occurring Professions:")
    for soc_code, count, risk_score, label in node_info[:10]:
        print(f"{label}: {count} occurrences, Risk Score: {risk_score:.2f}")
    
    print("\nTop 10 Paths with Highest Median Risk Scores:")
    for soc_code, median_risk, path_length, label, _ in path_medians[:10]:
        print(f"{label}: Median Risk: {median_risk:.2f}, Path Length: {path_length}")
    
    print("\nTop 10 Paths with Highest Mean Risk Scores:")
    for soc_code, mean_risk, path_length, label, _ in mean_scores[:10]:
        print(f"{label}: Mean Risk: {mean_risk:.2f}, Path Length: {path_length}")
