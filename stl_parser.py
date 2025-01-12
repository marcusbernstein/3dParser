from typing import List, Tuple
from collections import defaultdict

def parse(file_path: str, decimal_places: int = 5) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int, Tuple[float, float, float]]], List[Tuple[int, int]]]:
    unique_vertices = {}
    vertex_list = []
    triangles = []
    edge_to_normals = defaultdict(list)
    vertex_index = 0
    
    degenerate_count = 0
    
    with open(file_path, 'r') as file:
        current_triangle = []
        current_normal = None
        line_number = 0
        
        for line in file:
            line_number += 1
            stripped_line = line.strip()
            
            if stripped_line.startswith("facet normal"):
                parts = stripped_line.split()
                try:
                    current_normal = (
                        round(float(parts[2]), decimal_places),
                        round(float(parts[3]), decimal_places),
                        round(float(parts[4]), decimal_places)
                    )
                except (IndexError, ValueError) as e:
                    print(f"Warning: Invalid normal at line {line_number}: {stripped_line}")
                    current_normal = (0.0, 0.0, 0.0)
                
            elif stripped_line.startswith("vertex"):
                parts = stripped_line.split()
                try:
                    vertex = (
                        round(float(parts[1]), decimal_places),
                        round(float(parts[2]), decimal_places),
                        round(float(parts[3]), decimal_places)
                    )
                except (IndexError, ValueError) as e:
                    print(f"Warning: Invalid vertex at line {line_number}: {stripped_line}")
                    continue
                
                if vertex not in unique_vertices:
                    unique_vertices[vertex] = vertex_index
                    vertex_list.append(vertex)
                    vertex_index += 1
                    
                current_triangle.append(unique_vertices[vertex])
                
            elif stripped_line.startswith("endloop"):
                if len(current_triangle) == 3:
                    # Check for degenerate triangle
                    v1, v2, v3 = current_triangle
                    if v1 == v2 or v2 == v3 or v3 == v1:
                        print(f"Warning: Degenerate triangle found at line {line_number}: vertices {current_triangle}")
                        print(f"v1 {vertex_list[v1]} v2 {vertex_list[v2]} v3 {vertex_list[v3]}")

                        degenerate_count += 1
                    else:
                        # Add valid triangle with associated normal
                        triangles.append((v1, v2, v3, current_normal))
                        
                        # Add edges with their associated normal
                        # Only add edges between distinct vertices
                        for i in range(3):
                            v_start = current_triangle[i]
                            v_end = current_triangle[(i + 1) % 3]
                            if v_start != v_end:  # Only create edge if vertices are different
                                edge = tuple(sorted([v_start, v_end]))
                                edge_to_normals[edge].append(current_normal)
                
                current_triangle = []
    
    # Filter edges based on normal vectors
    final_edges = []
    for edge, normals in edge_to_normals.items():
        # If an edge has only one normal associated with it, it's a boundary edge
        # Or if it has multiple different normals, it's a boundary edge
        if len(normals) == 1 or any(n != normals[0] for n in normals):
            final_edges.append(edge)
    
    print(f"Statistics:")
    print(f"- Total vertices: {len(vertex_list)}")
    print(f"- Valid triangles: {len(triangles)}")
    print(f"- Degenerate triangles skipped: {degenerate_count}")
    print(f"- Boundary edges: {len(final_edges)}")
    
    return vertex_list, triangles, final_edges