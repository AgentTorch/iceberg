import os
import json
import networkx as nx
import csv
import pandas as pd

SCRAPED_DIR = os.path.join(os.path.dirname(__file__), '../rag/wynm_data_scrape/scraped_jsons')
RISK_CSV_PATH = os.path.join(os.path.dirname(__file__), '../v2_assets/skills_based_risk.csv')

def standardize_scores(scores):
    """Standardize scores to 0-100 range using min-max normalization."""
    min_score = min(scores.values())
    max_score = max(scores.values())
    return {k: ((v - min_score) / (max_score - min_score)) * 100 for k, v in scores.items()}

def build_graph():
    G = nx.DiGraph()
    name_to_soc = {}
    soc_to_name = {}
    soc_to_risk = {}

    # Load risk scores from CSV
    with open(RISK_CSV_PATH, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            soc_code = row['O*NET-SOC Code']
            risk_score = row.get('automation_risk_score')
            if soc_code and risk_score:
                try:
                    soc_to_risk[soc_code] = float(risk_score)
                except ValueError:
                    continue

    # Standardize risk scores
    soc_to_risk_standardized = standardize_scores(soc_to_risk)

    # Build graph nodes
    for fname in os.listdir(SCRAPED_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(SCRAPED_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                soc = data['summary'].get('soc_code')
                name = data['summary'].get('name')
                if soc and name:
                    job_title = name.split(' - ')[-1].strip()
                    name_to_soc[job_title] = soc
                    soc_to_name[soc] = job_title
                    # Set risk score to 0 if not found in the CSV
                    risk_score = soc_to_risk_standardized.get(soc, 0.0)
                    G.add_node(soc, label=job_title, automation_risk_score=risk_score)

    # Add edges from RelatedOccupations
    for fname in os.listdir(SCRAPED_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(SCRAPED_DIR, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                soc = data['summary'].get('soc_code')
                related = data['summary'].get('RelatedOccupations', [])
                for rel in related:
                    rel_soc = name_to_soc.get(rel.strip())
                    if soc and rel_soc:
                        G.add_edge(soc, rel_soc, relation='related')

    return G

def find_furthest_path_with_branches(G, start_node, num_branches=1):
    if start_node not in G:
        raise ValueError(f"Node {start_node} not in graph.")

    all_paths = nx.single_source_shortest_path(G, start_node)
    furthest_node = max(all_paths, key=lambda k: len(all_paths[k]))
    main_path = all_paths[furthest_node]

    branches = {}
    branch_paths = []
    main_path_set = set(main_path)
    used_nodes = set(main_path)

    # For each node in the main path (except the last), find all unique branches
    for i, node in enumerate(main_path[:-1]):
        next_node = main_path[i + 1]
        all_successors = set(G.successors(node))
        other_branches = all_successors - {next_node}
        branches[node] = list(other_branches)
        for branch_start in other_branches:
            # Find the longest path from this branch_start, avoiding overlap with main_path and other branches
            branch_all_paths = nx.single_source_shortest_path(G, branch_start)
            # Only consider paths that do not overlap with used_nodes
            valid_paths = [p for p in branch_all_paths.values() if not any(n in used_nodes for n in p[1:])]
            if valid_paths:
                longest = max(valid_paths, key=len)
                branch_paths.append(longest)
                used_nodes.update(longest)

    # Sort and select the specified number of longest unique branches
    branch_paths = sorted(branch_paths, key=len, reverse=True)[:num_branches]

    return {
        'start': start_node,
        'furthest': furthest_node,
        'path': main_path,
        'branches': branches,
        'longest_branches': branch_paths
    }

def build_paths_for_threshold(G, threshold, num_branches=1, output_csv=None):
    """
    Build paths for all nodes with automation risk score above the threshold.
    
    Args:
        G: NetworkX DiGraph
        threshold: Minimum automation risk score (0-100)
        num_branches: Number of branches to include for each path
        output_csv: Path to output CSV file (optional)
    
    Returns:
        List of dictionaries containing path information
    """
    high_risk_nodes = [node for node, data in G.nodes(data=True) 
                      if data.get('automation_risk_score', 0) >= threshold]
    
    all_paths = []
    for node in high_risk_nodes:
        try:
            path_info = find_furthest_path_with_branches(G, node, num_branches)
            path_info['start_risk_score'] = G.nodes[node]['automation_risk_score']
            path_info['start_label'] = G.nodes[node]['label']
            all_paths.append(path_info)
        except ValueError as e:
            print(f"Error processing node {node}: {e}")
            continue
    
    if output_csv:
        # Convert paths to DataFrame
        rows = []
        for path_info in all_paths:
            # Main path
            main_path = path_info['path']
            main_path_labels = [G.nodes[n]['label'] for n in main_path]
            main_path_scores = [G.nodes[n]['automation_risk_score'] for n in main_path]
            
            row = {
                'start_node': path_info['start'],
                'start_label': path_info['start_label'],
                'start_risk_score': path_info['start_risk_score'],
                'furthest_node': path_info['furthest'],
                'furthest_label': G.nodes[path_info['furthest']]['label'],
                'main_path': ' -> '.join(main_path_labels),
                'main_path_scores': ' -> '.join(map(str, main_path_scores))
            }
            
            # Add branch information
            for i, branch in enumerate(path_info['longest_branches']):
                branch_labels = [G.nodes[n]['label'] for n in branch]
                branch_scores = [G.nodes[n]['automation_risk_score'] for n in branch]
                row[f'branch_{i+1}'] = ' -> '.join(branch_labels)
                row[f'branch_{i+1}_scores'] = ' -> '.join(map(str, branch_scores))
            
            rows.append(row)
        
        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(rows)} paths to {output_csv}")
    
    return all_paths

if __name__ == "__main__":
    G = build_graph()
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Save graph
    nx.write_graphml(G, os.path.join(os.path.dirname(__file__), 'occupations_graph.graphml'))
    with open(os.path.join(os.path.dirname(__file__), 'occupations_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(nx.readwrite.json_graph.node_link_data(G, edges="edges"), f, indent=2)
    
    # Build paths for nodes with risk score >= 60
    output_csv = os.path.join(os.path.dirname(__file__), 'high_risk_paths.csv')
    paths = build_paths_for_threshold(G, threshold=60, num_branches=3, output_csv=output_csv)
    print(f"Found {len(paths)} paths for nodes with risk score >= 60")
