import pygame
import numpy as np
from constants import *

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


def render_shapes(screen, vertices, surfaces, recognized_contours, recognized_circles, scale, offset, rotation_matrix):
    for surface_index, surface in enumerate(surfaces):        
        # Get and rotate surface vertices
        surface_vertices = [vertices[i] for i in surface]
        rotated_surface_vertices = [np.dot(rotation_matrix, v) for v in surface_vertices]
        
        for contour in recognized_contours[surface_index]: # Render contours

            # Transform contour points to screen space
            transformed_points = []
            for point in contour['points']:
                z = vertices[surface[0]][2]  # Use Z from first vertex in surface
                point_3d = np.array([point[0], point[1], z])
                
                # Apply rotation
                rotated_point = np.dot(rotation_matrix, point_3d)
                
                # Project to screen space
                screen_x = int(rotated_point[0] * scale + offset[0])
                screen_y = int(-rotated_point[2] * scale + offset[1])
                transformed_points.append((screen_x, screen_y))
            
            if len(transformed_points) > 1:
                color = (255, 0, 0) if contour['is_hole'] else (0, 255, 0)
                pygame.draw.lines(screen, color, True, transformed_points, 2)
        
        for circle in recognized_circles[surface_index]: # Render circles

            x, y, radius = circle
            # Calculate circle center point using rotated surface vertices
            center_x = sum(v[0] for v in rotated_surface_vertices) / len(rotated_surface_vertices)
            center_y = sum(v[1] for v in rotated_surface_vertices) / len(rotated_surface_vertices)
            center_z = sum(v[2] for v in rotated_surface_vertices) / len(rotated_surface_vertices)
            
            # Project circle center
            projected_center_x = int(center_x * scale + offset[0])
            projected_center_y = int(-center_z * scale + offset[1])
            
            # Calculate ellipse dimensions considering rotation and scale
            normal_vector = [0, 0, 1]  # Z-axis normal vector
            rotated_normal_vector = np.dot(rotation_matrix, normal_vector)
            ellipse_major_axis = radius * scale
            ellipse_minor_axis = radius * scale * abs(rotated_normal_vector[1])
            
            # Draw projected ellipse
            pygame.draw.ellipse(screen, (0, 0, 255), 
                              (projected_center_x - ellipse_major_axis, 
                               projected_center_y - ellipse_minor_axis, 
                               2 * ellipse_major_axis, 
                               2 * ellipse_minor_axis), 2)