from typing import List, Tuple
from collections import defaultdict

def parse(file_path: str, decimal_places: int = 3) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    unique_vertices = {}
    vertex_list = []
    triangles = []
    # Dictionary to store edges and their associated triangle normals
    edge_to_normals = defaultdict(list)
    vertex_index = 0
    
    with open(file_path, 'r') as file:
        current_triangle = []
        current_normal = None
        
        for line in file:
            stripped_line = line.strip()
            
            # Parse normal vector
            if stripped_line.startswith("facet normal"):
                parts = stripped_line.split()
                current_normal = (
                    round(float(parts[2]), decimal_places),
                    round(float(parts[3]), decimal_places),
                    round(float(parts[4]), decimal_places)
                )
                
            elif stripped_line.startswith("vertex"):
                parts = stripped_line.split()
                vertex = (
                    round(float(parts[1]), decimal_places),
                    round(float(parts[2]), decimal_places),
                    round(float(parts[3]), decimal_places)
                )
                
                if vertex not in unique_vertices:
                    unique_vertices[vertex] = vertex_index
                    vertex_list.append(vertex)
                    vertex_index += 1
                    
                current_triangle.append(unique_vertices[vertex])
                
            elif stripped_line.startswith("endloop"):
                if len(current_triangle) == 3:
                    triangles.append(tuple(current_triangle))
                    
                    # Add edges with their associated normal
                    for i in range(3):
                        edge = tuple(sorted([current_triangle[i], current_triangle[(i + 1) % 3]]))
                        edge_to_normals[edge].append(current_normal)
                        
                current_triangle = []
    
    # Filter edges based on normal vectors
    final_edges = []
    for edge, normals in edge_to_normals.items():
        # If an edge has only one normal associated with it, it's a boundary edge
        # Or if it has multiple different normals, it's a boundary edge
        if len(normals) == 1 or any(n != normals[0] for n in normals):
            final_edges.append(edge)
    
    return vertex_list, triangles, final_edges