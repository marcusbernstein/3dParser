import numpy as np
from typing import Tuple

def parse(file_path):
    vertices = []
    triangles = []

    with open(file_path, 'r') as file:
        current_triangle = []
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("vertex"):
                parts = stripped_line.split()
                vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
                vertices.append(vertex)
                current_triangle.append(len(vertices) - 1)  # Store vertex index
            elif stripped_line.startswith("endloop"):
                if len(current_triangle) == 3:
                    triangles.append(current_triangle)
                current_triangle = []

    return np.array(vertices), triangles
