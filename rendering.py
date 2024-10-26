import pygame
import numpy as np

def render_stl(screen, vertices, triangles, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    
    for vertex in projected_vertices:
        pygame.draw.circle(screen, (255, 255, 255), vertex, 2)
    
    for triangle in triangles:
        v1, v2, v3 = triangle
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v1], projected_vertices[v2], 2)
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v2], projected_vertices[v3], 2)
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v3], projected_vertices[v1], 2)

def render_sketch_planes(screen, vertices, sketch_planes, scale, offset, rotation_matrix, colors, padding=5):
    for i, (axis, plane_magnitude, covered_vertices) in enumerate(sketch_planes):
        color = colors[i]
        # Extract the vertices on this sketch plane
        plane_vertices = np.array([vertices[v] for v in covered_vertices])
        
        min_coords = np.min(plane_vertices, axis=0)
        max_coords = np.max(plane_vertices, axis=0)
        center = (min_coords + max_coords) / 2

        # Add padding
        min_coords -= padding
        max_coords += padding

        # Create a rectangle based on the axis of the sketch plane
        if axis == 'x':
            rectangle = [
                [plane_magnitude, min_coords[1], min_coords[2]],  
                [plane_magnitude, max_coords[1], min_coords[2]],  
                [plane_magnitude, max_coords[1], max_coords[2]],  
                [plane_magnitude, min_coords[1], max_coords[2]],  
                [plane_magnitude, center[1], center[2]]  
            ]
        elif axis == 'y':
            rectangle = [
                [min_coords[0], plane_magnitude, min_coords[2]],  
                [max_coords[0], plane_magnitude, min_coords[2]],  
                [max_coords[0], plane_magnitude, max_coords[2]],  
                [min_coords[0], plane_magnitude, max_coords[2]],  
                [center[0], plane_magnitude, center[2]]  
            ]
        elif axis == 'z':
            rectangle = [
                [min_coords[0], min_coords[1], plane_magnitude],  
                [max_coords[0], min_coords[1], plane_magnitude],  
                [max_coords[0], max_coords[1], plane_magnitude],  
                [min_coords[0], max_coords[1], plane_magnitude],  
                [center[0], center[1], plane_magnitude]  
            ]

        rotated_rectangle = [np.dot(rotation_matrix, point) for point in rectangle]
        projected_rectangle = [(int(point[0] * scale + offset[0]), int(-point[2] * scale + offset[1]))
                               for point in rotated_rectangle]

        pygame.draw.line(screen, color, projected_rectangle[0], projected_rectangle[1], 2)
        pygame.draw.line(screen, color, projected_rectangle[1], projected_rectangle[2], 2)
        pygame.draw.line(screen, color, projected_rectangle[2], projected_rectangle[3], 2)
        pygame.draw.line(screen, color, projected_rectangle[3], projected_rectangle[0], 2)
        pygame.draw.circle(screen, color, projected_rectangle[4], 3)

def render_sketch_contours(screen, vertices, sketch_plane_edges, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]

    for plane in sketch_plane_edges:
        axis, magnitude, edges = plane
        for edge in edges:
            v1, v2 = edge
            if 0 <= v1 < len(projected_vertices) and 0 <= v2 < len(projected_vertices):
                pygame.draw.line(screen, (0, 255, 255), projected_vertices[v1], projected_vertices[v2], 2)

def render_edges(screen, vertices, edges, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    
    for edge in edges:
        v1, v2 = edge
        pygame.draw.line(screen, (0, 255, 255), projected_vertices[v1], projected_vertices[v2], 1)
        
def render_extrudes(screen, extrude_data, vertices, scale, offset, rotation_matrix, colors):
    for i, extrude in enumerate(extrude_data['extrudes']):
        color = colors[i]

        start_vertices = [vertices[idx] for idx, _ in extrude['matching_vertices']]
        end_vertices = [vertices[idx] for _, idx in extrude['matching_vertices']]

        start_center = np.mean(start_vertices, axis=0)
        end_center = np.mean(end_vertices, axis=0)

        rotated_start = np.dot(rotation_matrix, start_center)
        rotated_end = np.dot(rotation_matrix, end_center)

        projected_start = (int(rotated_start[0] * scale + offset[0]), int(-rotated_start[2] * scale + offset[1]))
        projected_end = (int(rotated_end[0] * scale + offset[0]), int(-rotated_end[2] * scale + offset[1]))

        pygame.draw.line(screen, color, projected_start, projected_end, 2)
        pygame.draw.circle(screen, color, projected_start, 4)
        pygame.draw.circle(screen, color, projected_end, 4)

def render_feature_tree(screen, extrude_data, colors, start_x=20, start_y=20):
    try:
        pygame.font.init()
        font = pygame.font.SysFont(None, 24) 
    except:
        print("Warning: Font initialization failed")
        return

    current_y = start_y
    line_height = 30  
    
    for i, extrude in enumerate(extrude_data['extrudes']):
        color = colors[i % len(colors)]
        start_plane = extrude['start_plane']
        end_plane = extrude['end_plane']
        sketch_number = i + 1  

        feature_text = f"sk{sketch_number} > Extrude {i + 1} (Start: {start_plane}, End: {end_plane})"
        text_surface = font.render(feature_text, True, color)
        screen.blit(text_surface, (start_x, current_y))

        current_y += line_height

def create_color_scheme(num_features):
    base_colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        #(0, 255, 255),    # Cyan
        (255, 182, 193),  # Light pink
        (152, 251, 152),  # Pale green
        (173, 216, 230),  # Light blue
        (210, 105, 30),   # Chocolate
        (184, 134, 11),   # Dark goldenrod
        (85, 107, 47),    # Dark olive green
        (255, 140, 0),    # Dark orange
        (148, 0, 211),    # Dark violet
        (0, 206, 209),    # Dark turquoise
        (219, 112, 147),  # Pale violet red
        (107, 142, 35),   # Olive drab
        (100, 149, 237),  # Cornflower blue
        (255, 20, 147),   # Deep pink
        (50, 205, 50),    # Lime green
        (25, 25, 112),    # Midnight blue
        (218, 165, 32),   # Golden rod
        (128, 0, 128),    # Purple
        (72, 209, 204),   # Medium turquoise
        (188, 143, 143)   # Rosy brown
    ]
    return [base_colors[i % len(base_colors)] for i in range(num_features)]