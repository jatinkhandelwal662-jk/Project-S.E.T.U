import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import math

def get_trajectory_vector(G, node):
    """
    Calculates the normalized directional vector of a road endpoint.
    Finds the single neighbor of the endpoint and points the vector outward.
    """
    neighbor = list(G.neighbors(node))[0]
    dy = node[0] - neighbor[0]
    dx = node[1] - neighbor[1]
    
    magnitude = math.hypot(dy, dx)
    if magnitude == 0: 
        return (0, 0)
    return (dy / magnitude, dx / magnitude)

def extract_criticality_from_mask(binary_mask, max_bridge_distance=60):
    """
    Phase II: Graph Skeletonization & Topological Healing 
    Fulfills ISRO requirement: MST and Disjoint Set algorithms based on 
    Euclidean distance AND angular alignment trajectory.
    """
    # 1. Thin the roads down to 1-pixel wide centerlines
    skeleton = skeletonize(binary_mask)
    
    G = nx.Graph()
    y_coords, x_coords = np.nonzero(skeleton)
    points = list(zip(y_coords, x_coords))
    
    if not points:
        return G, {}

    point_set = set(points)
    for p in points:
        G.add_node(p)
        
    # 2. Build the initial fragmented graph
    for y, x in points:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = (y + dy, x + dx)
                if neighbor in point_set:
                    G.add_edge((y, x), neighbor, weight=1.0, type='actual')

    # 3. Find the "Endpoints" of the broken fragments (degree == 1)
    endpoints = [node for node, degree in dict(G.degree()).items() if degree == 1]
    
    # 4. Topological Healing: Angular Alignment + Distance
    candidate_graph = G.copy()
    
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            p1 = endpoints[i]
            p2 = endpoints[j]
            
            # Calculate Euclidean Distance
            dy_b = p2[0] - p1[0]
            dx_b = p2[1] - p1[1]
            dist = math.hypot(dy_b, dx_b)
            
            if dist <= max_bridge_distance:
                vb1 = (dy_b / dist, dx_b / dist)  
                vb2 = (-dy_b / dist, -dx_b / dist) 
                
                v1 = get_trajectory_vector(G, p1)
                v2 = get_trajectory_vector(G, p2)
                
                # Dot Product for Angular Alignment (Cosine Similarity)
                # 90 degrees = 0.0, 0 degrees = 1.0
                align1 = (v1[0] * vb1[0]) + (v1[1] * vb1[1])
                align2 = (v2[0] * vb2[0]) + (v2[1] * vb2[1])
                
                # ISRO Constraint: Reject strict 90-degree zigzags (artifacts)
                # > 0.25 (approx 75 degrees) allows natural city side streets to connect
                # perfectly without breaking the ISRO rule against 90-deg angles.
                if align1 > 0.25 and align2 > 0.25:
                    
                    alignment_penalty = 2.0 - ((align1 + align2) / 2.0) 
                    weight = dist * alignment_penalty
                    
                    candidate_graph.add_edge(p1, p2, weight=weight, type='healed')

    # 5. Apply Minimum Spanning Tree (MST) via Disjoint Sets
    healed_graph = nx.minimum_spanning_tree(candidate_graph, weight='weight')
    
    # Clean up isolated nodes
    healed_graph.remove_nodes_from(list(nx.isolates(healed_graph)))

    # 6. Centrality Calculation (ISRO Node Ablation Requirement)
    centrality_scores = nx.betweenness_centrality(healed_graph, k=min(50, len(healed_graph.nodes())), weight='weight', seed=42)
    
    if centrality_scores:
        max_score = max(centrality_scores.values())
        if max_score > 0:
            for node in centrality_scores:
                centrality_scores[node] = centrality_scores[node] / max_score
                
    return healed_graph, centrality_scores


def calculate_impact_metrics(G_original, G_current):
    """
    Quantifies the systemic impact of infrastructure failure using Stochastic Sampling.
    Optimized for 10,000+ node graphs to prevent CPU timeout.
    """
    def get_sampled_centrality(G):
        # Seed fixed at 42 to ensure consistent comparison before/after ablation
        centrality = nx.betweenness_centrality(G, k=min(50, len(G.nodes())), weight='weight', seed=42)
        return sum(centrality.values()) if centrality else 0.0

    c_original = get_sampled_centrality(G_original)
    c_current = get_sampled_centrality(G_current)
    
    resilience_index = 100.0
    if c_original > 0:
        resilience_index = round((c_current / c_original) * 100, 2)
        resilience_index = min(100.0, max(0.0, resilience_index))

    impact_delta = 0.0
    if resilience_index < 100.0:
        impact_delta = round(100.0 - resilience_index, 2)
        
    return resilience_index, impact_delta
