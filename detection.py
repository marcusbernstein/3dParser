import numpy as np
import cv2
from typing import List, Tuple


def detect_flat_surfaces(vertices, triangles, tolerance=1e-6):
    coplanar_triangles = {} # Identify triangles with same dimension (X, Y, or Z) for all vertices
    for tri in triangles:
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        for dim in range(3):
            if abs(v1[dim] - v2[dim]) < tolerance and abs(v2[dim] - v3[dim]) < tolerance:
                if dim not in coplanar_triangles:
                    coplanar_triangles[dim] = {}
                dim_value = round(v1[dim], 6)
                if dim_value not in coplanar_triangles[dim]:
                    coplanar_triangles[dim][dim_value] = []
                coplanar_triangles[dim][dim_value].append(tri)

    # Identify flat surfaces
    flat_surfaces = []
    for dim, dim_values in coplanar_triangles.items():
        for dim_value, triangles in dim_values.items():
            if len(triangles) > 0:
                flat_surface = []
                for tri in triangles:
                    flat_surface.extend(tri)
                flat_surfaces.append(list(set(flat_surface)))

    return flat_surfaces

def recognize_shapes(vertices, edges, surfaces):
    recognized_contours = []
    recognized_circles = []

    for surface_index, surface in enumerate(surfaces):
        surface_vertices_3d = [vertices[i] for i in surface]
        
        # Calculate surface normal
        v0 = np.array(surface_vertices_3d[0])
        v1 = np.array(surface_vertices_3d[1])
        v2 = np.array(surface_vertices_3d[2])
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        
        # Find the best projection plane (xy, yz, or xz)
        abs_normal = np.abs(normal)
        max_component = np.argmax(abs_normal)
        
        surface_vertices_2d = [] # Project vertices onto the appropriate plane
        for vertex in surface_vertices_3d:
            if max_component == 0:  # Project onto yz plane
                surface_vertices_2d.append((vertex[1], vertex[2]))
            elif max_component == 1:  # Project onto xz plane
                surface_vertices_2d.append((vertex[0], vertex[2]))
            else:  # Project onto xy plane
                surface_vertices_2d.append((vertex[0], vertex[1]))
                
        surface_vertices_2d = np.array(surface_vertices_2d)
        
        surface_edges = [edge for edge in edges if edge[0] in surface and edge[1] in surface]
        
        image_size = 1200
        padding = 50
        
        min_x = np.min(surface_vertices_2d[:, 0]) - padding
        min_y = np.min(surface_vertices_2d[:, 1]) - padding
        max_x = np.max(surface_vertices_2d[:, 0]) + padding
        max_y = np.max(surface_vertices_2d[:, 1]) + padding
        
        width = max_x - min_x
        height = max_y - min_y
        if width == 0 or height == 0:
            continue
            
        scale = min(image_size / width, image_size / height) * 0.9
        
        image = np.zeros((image_size, image_size), np.uint8)
        
        transform = {
            'scale': scale,
            'min_x': min_x,
            'min_y': min_y,
            'max_component': max_component,
            'image_size': image_size
        }
        
        for edge in surface_edges:
            v1 = surface_vertices_3d[surface.index(edge[0])]
            v2 = surface_vertices_3d[surface.index(edge[1])]
            
            if max_component == 0:
                p1 = (v1[1], v1[2])
                p2 = (v2[1], v2[2])
            elif max_component == 1:
                p1 = (v1[0], v1[2])
                p2 = (v2[0], v2[2])
            else:
                p1 = (v1[0], v1[1])
                p2 = (v2[0], v2[1])
                
            x1 = int((p1[0] - min_x) * scale)
            y1 = int((p1[1] - min_y) * scale)
            x2 = int((p2[0] - min_x) * scale)
            y2 = int((p2[1] - min_y) * scale)
            
            cv2.line(image, (x1, y1), (x2, y2), 255, 2)
        
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        surface_contours = []
        surface_circles = []
        
        if len(contours) > 0 and hierarchy is not None:
            for idx, (contour, h) in enumerate(zip(contours, hierarchy[0])):
                if len(contour) < 4:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                if area < 100: 
                    continue
                
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity > 0.85: # If circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    # Transform back to 3D space
                    orig_x = x / scale + min_x
                    orig_y = y / scale + min_y
                    orig_radius = radius / scale
                    surface_circles.append((orig_x, orig_y, orig_radius))
                else:  # If polygon
                    epsilon = 0.02 * perimeter # Approximate the contour to reduce noise
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) > 3: # I hate triangless!
                        contour_points = []
                        for point in approx:
                            x, y = point[0]
                            orig_x = x / scale + min_x
                            orig_y = y / scale + min_y
                            
                            if max_component == 0:
                                point_3d = (vertices[surface[0]][0], orig_x, orig_y)
                            elif max_component == 1:
                                point_3d = (orig_x, vertices[surface[0]][1], orig_y)
                            else:
                                point_3d = (orig_x, orig_y, vertices[surface[0]][2])
                            
                            contour_points.append((point_3d[0], point_3d[1]))
                        
                        surface_contours.append({ # Store hierarchy information with the contour
                            'points': contour_points,
                            'parent': h[3],  # Parent contour index
                            'is_hole': h[3] != -1  # True if this contour has a parent
                        })
        
        recognized_contours.append(surface_contours)
        recognized_circles.append(surface_circles)
        
    return recognized_contours, recognized_circles

def transform_point(x, y, transform):
    scale = transform['scale']
    center_x = transform['center_x']
    center_y = transform['center_y']
    image_size = transform['image_size']
    
    px = int((x - center_x) * scale + image_size // 2)
    py = int((y - center_y) * scale + image_size // 2)
    return (px, py)