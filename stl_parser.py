from typing import List, Tuple

def parse(file_path: str, decimal_places: int = 3) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    unique_vertices = {}
    vertex_list = []
    triangles = []
    edges = set()
    vertex_index = 0  # Keep track of index for new vertices

    with open(file_path, 'r') as file:
        current_triangle = []
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("vertex"):
                parts = stripped_line.split()
                # Parse and round the vertex coordinates
                vertex = (
                    round(float(parts[1]), decimal_places),
                    round(float(parts[2]), decimal_places),
                    round(float(parts[3]), decimal_places)
                )
                # If the vertex is new, add it to the unique vertices
                if vertex not in unique_vertices:
                    unique_vertices[vertex] = vertex_index
                    vertex_list.append(vertex)
                    vertex_index += 1
                # Add the index of the vertex to the current triangle
                current_triangle.append(unique_vertices[vertex])
            elif stripped_line.startswith("endloop"):
                if len(current_triangle) == 3:
                    # Add the triangle
                    triangles.append(tuple(current_triangle))
                    # Add the edges of the triangle
                    for i in range(3):
                        edge = tuple(sorted([current_triangle[i], current_triangle[(i + 1) % 3]]))
                        edges.add(edge)
                current_triangle = []  # Reset for the next triangle

    return vertex_list, triangles, list(edges)