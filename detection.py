import numpy as np
import matplotlib.pyplot as plt
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
    """
    Recognize shapes in the 3D model.

    Args:
    surfaces (List[List[int]]): Surface indices.
    vertices (List[Tuple[float, float, float]]): 3D model vertices.

    Returns:
    List[Tuple[str, List[float], int, Tuple[int, int]]]: Recognized shapes with type, dimensions, surface index, and contour coordinates.
    """

    recognized_shapes = []
    for surface_index, surface in enumerate(surfaces):
        print(f"Processing surface {surface_index}")
        # Project vertices to 2D (ignore z-axis)
        projected_vertices = [(vertices[i][0], vertices[i][1]) for i in surface]
        
        # Convert to numpy array for OpenCV
        projected_vertices = np.array(projected_vertices)
        
        # Check if there are at least 3 points
        if len(projected_vertices) < 3:
            continue
        
        # Create a blank image
        min_x, min_y = np.min(projected_vertices, axis=0)
        max_x, max_y = np.max(projected_vertices, axis=0)
        image_size = int(max(max_x - min_x, max_y - min_y) * 1.2)
        image = np.zeros((image_size, image_size), np.uint8)
        
        # Draw contours
        offset = (image_size // 2, image_size // 2)
        translated_vertices = (projected_vertices - [min_x, min_y]) + offset
        translated_vertices = translated_vertices.astype(np.int32)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
        
        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1:
                continue
            
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) > 7:
                shape_type = "Circle"
            elif (len(approx) <= 7 and len(approx) >4) :
                shape_type = "Polygon"
            elif len(approx) == 4:
                shape_type = "Quad"
            elif len(approx) == 3:
                shape_type = "Tri"
            else:
                shape_type = "Unknown"
            
            x, y, w, h = cv2.boundingRect(contour)
            dimensions = [w/100, h/100]
            
            # Get contour coordinates
            contour_coords = tuple(map(tuple, contour.squeeze().tolist()))
            
            recognized_shapes.append((shape_type, dimensions, surface_index, contour_coords))
            print(f"Shape identified: {shape_type} on surface {surface_index} with dimensions {dimensions} at contour {contour_coords}")
    
    print(f"Total shapes identified: {len(recognized_shapes)}")
    
    return recognized_shapes

"""def track_extrudes(recognized_shapes, tolerance=0.01):
    ""
    Track recognized shapes across flat surfaces to detect extrusions.

    Args:
    recognized_shapes (List[Tuple[str, List[float], int]]): Recognized shapes with type, dimensions, and surface index.
    tolerance (float, optional): Tolerance for shape matching. Defaults to 0.01.

    Returns:
    List[Tuple[str, List[int]]]: Identified 3D shapes with their surface indices.
    ""

    # Initialize a list to store identified 3D shapes
    identified_3d_shapes = []

    # Iterate over recognized shape types
    shape_types = set(shape[0] for shape in recognized_shapes)
    for shape_type in shape_types:
        # Filter shapes by type
        type_shapes = [shape for shape in recognized_shapes if shape[0] == shape_type]

        # Group surfaces by their dimensions
        dimension_groups = {}
        for shape in type_shapes:
            dimensions = tuple(shape[1])  # Use tuple to make dimensions hashable
            surface = shape[2]
            if dimensions not in dimension_groups:
                dimension_groups[dimensions] = []
            dimension_groups[dimensions].append(surface)

        # Identify 3D shape for each dimension group
        for dimensions, surfaces in dimension_groups.items():
            if len(surfaces) > 1:
                if shape_type == "Circle":
                    identified_3d_shapes.append(("Cylinder", surfaces))
                elif shape_type == "Triangle":
                    identified_3d_shapes.append(("Prism", surfaces))
                elif shape_type == "Quad":
                    identified_3d_shapes.append(("Box", surfaces))
                else:
                    identified_3d_shapes.append((f"Extruded {shape_type}", surfaces))

    print("Identified 3D Shapes:")
    for shape_type, surfaces in identified_3d_shapes:
        print(f"{shape_type} found on {len(surfaces)} surfaces")

    return identified_3d_shapes"""