import os
import networkx as nx
import matplotlib.pyplot as plt
from analyze_graph import analyze_high_risk_professions, analyze_path_medians, calculate_mean_risk_scores
from build_graph import build_graph
from visualize_graph import visualize_path_with_branches

# If any code loads JSONs, ensure it uses rag/wynm_data_scrape/scraped_jsons as the directory.
# (No explicit path found in this file, but add a comment for clarity.)

def visualize_top_risky_paths(threshold=60, num_branches=3, num_paths=5):
    """
    Analyze and visualize the top risky paths.
    
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
        num_paths: Number of top paths to visualize
    """
    # Get the graph
    G = build_graph()
    
    # Run frequency analysis
    node_info = analyze_high_risk_professions(threshold=threshold, num_branches=num_branches)
    
    # Print top num_paths most frequent professions
    print(f"\nTop {num_paths} Most Frequently Occurring Professions in High-Risk Paths:")
    print("=" * 80)
    print(f"{'Job Title':<50} {'Occurrences':<12} {'Risk Score':<12} {'SOC Code'}")
    print("-" * 80)
    for soc_code, count, risk_score, label in node_info[:num_paths]:
        print(f"{label:<50} {count:<12} {risk_score:<12.2f} {soc_code}")
    
    # Run median analysis
    path_medians = analyze_path_medians(threshold=threshold, num_branches=num_branches)
    
    # Print top num_paths paths with highest median risks
    print(f"\nTop {num_paths} Paths with Highest Median Risk Scores:")
    print("=" * 80)
    print(f"{'Job Title':<50} {'Median Risk':<12} {'Path Length':<12} {'SOC Code'}")
    print("-" * 80)
    for soc_code, median_risk, path_length, label, _ in path_medians[:num_paths]:
        print(f"{label:<50} {median_risk:<12.2f} {path_length:<12} {soc_code}")
    
    # Calculate mean risk scores for all paths
    all_paths = []
    for soc_code, _, _, _, path in path_medians:
        if path and isinstance(path, list):
            all_paths.append(path)
    
    mean_scores = calculate_mean_risk_scores(G, all_paths)
    
    # Print top num_paths paths with highest mean risk scores
    print(f"\nTop {num_paths} Paths with Highest Mean Risk Scores:")
    print("=" * 80)
    print(f"{'Job Title':<50} {'Mean Risk':<12} {'Path Length':<12} {'SOC Code'}")
    print("-" * 80)
    for soc_code, mean_risk, path_length, label, _ in mean_scores[:num_paths]:
        print(f"{label:<50} {mean_risk:<12.2f} {path_length:<12} {soc_code}")
    
    # Create output directory for visualizations
    output_dir = os.path.join(os.path.dirname(__file__), 'path_visualizations', 'visualize_high_risk')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each of the top paths
    for i, (soc_code, median_risk, path_length, label, _) in enumerate(path_medians[:num_paths]):
        print(f"\nVisualizing path {i+1}/{num_paths}:")
        print(f"Job: {label}")
        print(f"Median Risk: {median_risk:.2f}")
        print(f"Path Length: {path_length}")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        visualize_path_with_branches(G, soc_code, num_branches=num_branches)
        
        # Save the visualization
        output_file = os.path.join(output_dir, f'path_{i+1}_{soc_code}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {output_file}")

def visualize_overlap_paths(threshold=60, num_branches=3, num_paths=5):
    """
    Visualize the top N overlap (frequency) paths.
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
        num_paths: Number of top paths to visualize
    """
    G = build_graph()
    output_dir = os.path.join(os.path.dirname(__file__), 'path_visualizations', 'visualize_overlap')
    os.makedirs(output_dir, exist_ok=True)
    node_info = analyze_high_risk_professions(threshold=threshold, num_branches=num_branches)
    print(f"\nVisualizing top {num_paths} overlap (frequency) paths:")
    for i, (soc_code, count, risk_score, label) in enumerate(node_info[:num_paths]):
        print(f"Visualizing overlap path {i+1}/{num_paths}: {label}")
        plt.figure(figsize=(15, 10))
        visualize_path_with_branches(G, soc_code, num_branches=num_branches)
        output_file = os.path.join(output_dir, f'overlap_path_{i+1}_{soc_code}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {output_file}")

def visualize_median_paths(threshold=60, num_branches=3, num_paths=5):
    """
    Visualize the top N median risk paths.
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
        num_paths: Number of top paths to visualize
    """
    G = build_graph()
    output_dir = os.path.join(os.path.dirname(__file__), 'path_visualizations', 'visualize_median')
    os.makedirs(output_dir, exist_ok=True)
    path_medians = analyze_path_medians(threshold=threshold, num_branches=num_branches)
    print(f"\nVisualizing top {num_paths} median risk paths:")
    for i, (soc_code, median_risk, path_length, label, _) in enumerate(path_medians[:num_paths]):
        print(f"Visualizing median path {i+1}/{num_paths}: {label}")
        plt.figure(figsize=(15, 10))
        visualize_path_with_branches(G, soc_code, num_branches=num_branches)
        output_file = os.path.join(output_dir, f'median_path_{i+1}_{soc_code}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {output_file}")

def visualize_mean_paths(threshold=60, num_branches=3, num_paths=5):
    """
    Visualize the top N mean risk paths.
    Args:
        threshold: Minimum automation risk score to consider (0-100)
        num_branches: Number of branches to include for each path
        num_paths: Number of top paths to visualize
    """
    G = build_graph()
    output_dir = os.path.join(os.path.dirname(__file__), 'path_visualizations', 'visualize_mean')
    os.makedirs(output_dir, exist_ok=True)
    path_medians = analyze_path_medians(threshold=threshold, num_branches=num_branches)
    all_paths = []
    for soc_code, _, _, _, path in path_medians:
        if path and isinstance(path, list):
            all_paths.append(path)
    mean_scores = calculate_mean_risk_scores(G, all_paths)
    print(f"\nVisualizing top {num_paths} mean risk paths:")
    for i, (soc_code, mean_risk, path_length, label, _) in enumerate(mean_scores[:num_paths]):
        print(f"Visualizing mean path {i+1}/{num_paths}: {label}")
        plt.figure(figsize=(15, 10))
        visualize_path_with_branches(G, soc_code, num_branches=num_branches)
        output_file = os.path.join(output_dir, f'mean_path_{i+1}_{soc_code}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved visualization to {output_file}")

# Placeholder for future extreme path visualizations
def visualize_extreme_paths(*args, **kwargs):
    """
    Placeholder for visualizing extreme paths (e.g., highest/lowest risk, longest, etc.)
    """
    print("Extreme path visualization not yet implemented.")

if __name__ == "__main__":
    # Set parameters
    THRESHOLD = 70  # Minimum risk score to consider
    NUM_BRANCHES = 0  # Number of branches to show
    NUM_PATHS = 10  # Number of top paths to visualize
    
    # Visualize top paths
   
    visualize_overlap_paths(threshold=THRESHOLD, num_branches=NUM_BRANCHES, num_paths=NUM_PATHS)

    visualize_median_paths(threshold=THRESHOLD, num_branches=NUM_BRANCHES, num_paths=NUM_PATHS)

    visualize_mean_paths(threshold=THRESHOLD, num_branches=NUM_BRANCHES, num_paths=NUM_PATHS)

    visualize_top_risky_paths(threshold=THRESHOLD, num_branches=NUM_BRANCHES, num_paths=NUM_PATHS)

    
    print("\nAnalysis complete! Check the 'path_visualizations' directory for the generated visualizations.") 