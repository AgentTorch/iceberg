import networkx as nx
import matplotlib.pyplot as plt

def visualize_path_with_branches(G, start_node, num_branches=3):
    """
    Visualize only the main path and its branches with distinct colors for start and end nodes.
    
    Args:
        G: NetworkX graph
        start_node: Starting node (SOC code)
        num_branches: Number of branches to include
    """
    from build_graph import find_furthest_path_with_branches
    
    if start_node not in G:
        print(f"Start node {start_node} not in graph.")
        return
    
    # Get the path and its branches
    result = find_furthest_path_with_branches(G, start_node, num_branches)
    path = result['path']
    branches = result['longest_branches']

    # Collect all nodes and edges in the main path and branches
    nodes_to_draw = set(path)
    edges_to_draw = list(zip(path[:-1], path[1:]))
    for branch in branches:
        nodes_to_draw.update(branch)
        edges_to_draw.extend(list(zip(branch[:-1], branch[1:])))

    # Create a subgraph for visualization
    H = G.subgraph(nodes_to_draw).copy()
    pos = nx.spring_layout(H, k=1, iterations=50)

    # Draw all nodes in the subgraph as light gray (for background, optional)
    # nx.draw_networkx_nodes(H, pos, node_color='lightgray', node_size=500, alpha=0.2)

    # Draw all edges in the subgraph as light gray (for background, optional)
    # nx.draw_networkx_edges(H, pos, edge_color='lightgray', alpha=0.2)

    # Draw the main path nodes and edges
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(H, pos, edgelist=path_edges, edge_color='blue', width=2)
    nx.draw_networkx_nodes(H, pos, nodelist=path, node_color='blue', node_size=500)

    # Draw branch nodes and edges
    for branch in branches:
        branch_edges = list(zip(branch[:-1], branch[1:]))
        nx.draw_networkx_edges(H, pos, edgelist=branch_edges, edge_color='green', width=1.5)
        nx.draw_networkx_nodes(H, pos, nodelist=branch, node_color='green', node_size=400)

    # Highlight start and end nodes
    start_color = 'red'  # Start node in red
    end_color = 'purple'  # End node in purple
    nx.draw_networkx_nodes(H, pos, nodelist=[path[0]], node_color=start_color, node_size=700)
    nx.draw_networkx_nodes(H, pos, nodelist=[path[-1]], node_color=end_color, node_size=700)

    # Add labels with risk scores
    labels = {node: f"{H.nodes[node].get('label', node)}\n({H.nodes[node].get('automation_risk_score', 0):.1f})"
             for node in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels, font_size=8)

    # Add title
    plt.title(f"Career Path from {H.nodes[path[0]].get('label', path[0])}\n"
             f"Risk Score: {H.nodes[path[0]].get('automation_risk_score', 0):.1f}")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Start Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='End Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Main Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Branches')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')

if __name__ == "__main__":
    # Example usage
    G = nx.DiGraph()
    G.add_node("A", label="Node A", automation_risk_score=50.0)
    G.add_node("B", label="Node B", automation_risk_score=60.0)
    G.add_edge("A", "B")
    visualize_path_with_branches(G, "A", num_branches=1)
    plt.show()

# If any code loads JSONs, ensure it uses rag/wynm_data_scrape/scraped_jsons as the directory.
# (No explicit path found in this file, but add a comment for clarity.) 