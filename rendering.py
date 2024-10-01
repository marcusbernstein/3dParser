import pygame
import math
import numpy as np
from constants import *

def project_vertex(vertex, scale, offset):
    """Project 3D vertex to 2D screen coordinates."""
    x = vertex[0] * scale + offset[0]
    y = -vertex[2] * scale + offset[1]
    return (x, y)

def render_stl(screen, vertices, triangles, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    for vertex in projected_vertices:
        pygame.draw.circle(screen, WHITE, vertex, 2)
    for triangle in triangles:
        v1, v2, v3 = triangle
        pygame.draw.line(screen, WHITE, projected_vertices[v1], projected_vertices[v2], 2)
        pygame.draw.line(screen, WHITE, projected_vertices[v2], projected_vertices[v3], 2)
        pygame.draw.line(screen, WHITE, projected_vertices[v3], projected_vertices[v1], 2)
def render_shapes(screen, vertices, triangles, shapes, scale, offset, rotation_matrix):
    """Render recognized shapes on the screen."""
    for shape_type, surface in shapes:
        if shape_type == "Circle":
            # Calculate circle center point
            rotated_vertices = [np.dot(rotation_matrix, vertices[i]) for i in surface]
            center_x = sum(x for x, y, z in rotated_vertices) / len(rotated_vertices)
            center_y = sum(y for x, y, z in rotated_vertices) / len(rotated_vertices)
            center_z = sum(z for x, y, z in rotated_vertices) / len(rotated_vertices)

            # Project circle center
            projected_center_x = int(center_x * scale + offset[0])
            projected_center_y = int(-center_z * scale + offset[1])

            # Calculate ellipse dimensions
            radius = max(math.hypot(x - center_x, y - center_y) for x, y, z in rotated_vertices)
            normal_vector = [0, 0, 1]  # Z-axis normal vector
            rotated_normal_vector = np.dot(rotation_matrix, normal_vector)

            # Calculate ellipse axes
            ellipse_major_axis = radius * scale  # Constant width
            ellipse_minor_axis = radius * scale * math.fabs(rotated_normal_vector[1])  # Y-component controls height

            # Draw projected ellipse
            pygame.draw.ellipse(screen, (0, 255, 0), 
                                (projected_center_x - ellipse_major_axis, 
                                 projected_center_y - ellipse_minor_axis, 
                                 2 * ellipse_major_axis, 
                                 2 * ellipse_minor_axis), 2)
            
        elif shape_type == "Triangle":
            pygame.draw.polygon(screen, (255, 105, 180), projected_vertices, 2)
        elif shape_type == "Quadrilateral":
            min_x = min(x for x, y in projected_vertices)
            max_x = max(x for x, y in projected_vertices)
            min_y = min(y for x, y in projected_vertices)
            max_y = max(y for x, y in projected_vertices)
            pygame.draw.rect(screen, (255, 105, 180), (min_x, min_y, max_x - min_x, max_y - min_y), 2)
        elif shape_type == "Polygon":
            pygame.draw.polygon(screen, (255, 105, 180), projected_vertices, 2)