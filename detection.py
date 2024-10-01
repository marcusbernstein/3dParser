import numpy as np
import cv2
import math
import pygame
import itertools
from scipy.optimize import leastsq
from scipy.spatial import KDTree
from typing import List, Tuple
from collections import deque

def detect_flat_surfaces(vertices, triangles, tolerance=1e-6):
    """Detect flat surfaces from triangles."""
    print(f"Total triangles: {len(triangles)}")

    # Identify triangles with same dimension (X, Y, or Z) for all vertices
    coplanar_triangles = {}
    for tri in triangles:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Check if all vertices have the same value in any dimension
        for dim in range(3):
            if abs(v1[dim] - v2[dim]) < tolerance and abs(v2[dim] - v3[dim]) < tolerance:
                if dim not in coplanar_triangles:
                    coplanar_triangles[dim] = {}
                dim_value = round(v1[dim], 6)  # Round to 6 decimal places
                if dim_value not in coplanar_triangles[dim]:
                    coplanar_triangles[dim][dim_value] = []
                coplanar_triangles[dim][dim_value].append(tri)

    # Print coplanar triangles
    for dim, dim_values in coplanar_triangles.items():
        print(f"Coplanar triangles in dim {dim}:")
        for dim_value, triangles in dim_values.items():
            print(f"  Value {dim_value}: {len(triangles)} triangles")

    # Identify flat surfaces
    flat_surfaces = []
    for dim, dim_values in coplanar_triangles.items():
        for dim_value, triangles in dim_values.items():
            if len(triangles) > 0:  # Check for non-empty triangle list
                flat_surface = []
                for tri in triangles:
                    flat_surface.extend(tri)
                flat_surfaces.append(list(set(flat_surface)))

    print(f"Detected flat surfaces: {len(flat_surfaces)}")
    return flat_surfaces



def recognize_shapes(surfaces, vertices):
    recognized_shapes = []
    for surface in surfaces:
        shape_type = "Flat Surface"
                
        # Project vertices to 2D (ignore one axis)
        axis = 2  # Ignore z-axis for projection
        projected_vertices = [(v[0], v[1]) for v in vertices]
        
        # Convert to numpy array
        points_2d = np.array(projected_vertices, dtype=np.float32).reshape(-1, 1, 2)
        
        # Simplify polygon using approxPolyDP
        simplified_polygon = cv2.approxPolyDP(points_2d, epsilon=1, closed=True)
        
        # Determine shape type based on number of sides
        num_sides = len(simplified_polygon)
        if num_sides >=9:
            shape_type = "Circle"
        elif (num_sides >= 5 and num_sides < 9):
            shape_type = "Polygon"
        elif num_sides == 4:
            shape_type = "Quad"
        elif num_sides == 3:
            shape_type = "Tri"
        else:
            shape_type = "Unknown"
            print(shape_type)
        
        recognized_shapes.append((shape_type, surface))
    
    return recognized_shapes