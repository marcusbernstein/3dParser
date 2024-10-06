import pygame
import math
import numpy as np
from constants import *
from detection.py import *

def project_vertex(vertex, scale, offset):
    """Project 3D vertex to 2D screen coordinates."""
    x = vertex[0] * scale + offset[0]
    y = -vertex[2] * scale + offset[1]
    return (x, y)

def render_stl(screen, vertices, triangles, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    print("projected_vertices")
    print(projected_vertices)
    for vertex in projected_vertices:
        pygame.draw.circle(screen, WHITE, vertex, 2)
    for triangle in triangles:
        v1, v2, v3 = triangle
        pygame.draw.line(screen, WHITE, projected_vertices[v1], projected_vertices[v2], 2)
        pygame.draw.line(screen, WHITE, projected_vertices[v2], projected_vertices[v3], 2)
        pygame.draw.line(screen, WHITE, projected_vertices[v3], projected_vertices[v1], 2)
def render_shapes(screen, vertices, triangles, shapes, scale, offset, rotation_matrix):
    """
    Render recognized shapes on the screen.

    Args:
    screen: Pygame screen object.
    vertices: List[Tuple[float, float, float]]: 3D model vertices.
    triangles: List[List[int]]: 3D model triangles.
    shapes: List[Tuple[str, List[float], int, Tuple[int, int]]]: Recognized shapes.
    scale: float: Scaling factor.
    offset: Tuple[float, float]: Offset coordinates.
    rotation_matrix: numpy.ndarray: Rotation matrix.
    """

    # Render 3D model
    for triangle in triangles:
        point1 = np.dot(rotation_matrix, vertices[triangle[0]])
        point2 = np.dot(rotation_matrix, vertices[triangle[1]])
        point3 = np.dot(rotation_matrix, vertices[triangle[2]])
        
        x1, y1, _ = point1
        x2, y2, _ = point2
        x3, y3, _ = point3
        
        x1, y1 = int(x1 * scale + offset[0]), int(y1 * scale + offset[1])
        x2, y2 = int(x2 * scale + offset[0]), int(y2 * scale + offset[1])
        x3, y3 = int(x3 * scale + offset[0]), int(y3 * scale + offset[1])
        
        pygame.draw.polygon(screen, (255, 255, 255), [(x1, y1), (x2, y2), (x3, y3)], 1)

    # Calculate projected vertices
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]

    # Render recognized shapes
    for shape_type, dimensions, surface_index, contour_coords in shapes:
        color = (255, 0, 0)  # Red color for shapes
        
        # Project contour coordinates
        projected_contour = []
        for point in contour_coords:
            vertex_index = surface_index * 3 + point[1]
            vertex = vertices[vertex_index]
            projected_point = np.dot(rotation_matrix, vertex)
            x, y, _ = projected_point
            x, y = int(x * scale + offset[0]), int(y * scale + offset[1])
            projected_contour.append((x, y))
        
        pygame.draw.polygon(screen, color, projected_contour, 2)
        font = pygame.font.Font(None, 24)
        text = font.render(shape_type, True, color)
        screen.blit(text, (projected_contour[0][0], projected_contour[0][1] - 20))

    pygame.display.flip()