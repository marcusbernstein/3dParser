from collections import defaultdict
import numpy as np
import pygame

def transform_to_global(vertices, transformation_matrix):
    """Convert vertices from local to global coordinates using the provided transformation matrix."""
    global_vertices = []
    for vertex in vertices:
        homogenous_vertex = np.array([*vertex, 1])
        global_vertex = np.dot(np.linalg.inv(transformation_matrix), homogenous_vertex)
        global_vertices.append(global_vertex[:3])
    return global_vertices

def create_color_scheme(num_features):
    base_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (255, 182, 193), (152, 251, 152), (173, 216, 230),
        (210, 105, 30), (184, 134, 11), (85, 107, 47), (255, 140, 0),
        (148, 0, 211), (0, 206, 209), (219, 112, 147), (107, 142, 35),
        (100, 149, 237), (255, 20, 147), (50, 205, 50), (25, 25, 112),
        (218, 165, 32), (128, 0, 128), (72, 209, 204), (188, 143, 143)
    ]
    return [base_colors[i % len(base_colors)] for i in range(num_features)]

def render_stl(screen, vertices, triangles, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    
    for vertex in projected_vertices:
        pygame.draw.circle(screen, (255, 255, 255), vertex, 2)
    
    for triangle in triangles:
        v1, v2, v3, normal = triangle
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v1], projected_vertices[v2], 2)
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v2], projected_vertices[v3], 2)
        pygame.draw.line(screen, (255, 255, 255), projected_vertices[v3], projected_vertices[v1], 2)

def render_sketch_planes(screen, vertices, sketches, scale, offset, rotation_matrix, colors, padding=5):
    for sketch in sketches:
        color = colors[sketch['index']]
        normal = np.array(sketch['normal'])
        magnitude = sketch['magnitude']
        
        # Create an orthonormal basis for the sketch plane
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.array([0, -1, 0]) #if abs(normal[1]) > 0.9 else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        # Transform matrix from local sketch space to global space
        local_to_global = np.column_stack((x_axis, y_axis, z_axis))
        
        # Find bounds of sketch in local coordinates
        local_points = []
        for edge in sketch['edges']:
            start, end = edge
            local_points.extend([np.array([start[0], start[1], 0]), 
                               np.array([end[0], end[1], 0])])
        
        local_points = np.array(local_points)
        if len(local_points) == 0:
            continue
            
        min_bounds = np.min(local_points, axis=0) - padding
        max_bounds = np.max(local_points, axis=0) + padding
        
        # Generate corners of the sketch plane in local coordinates
        corners_local = [
            np.array([max_bounds[1], max_bounds[0], 0]),  # Bottom left
            np.array([min_bounds[1], max_bounds[0], 0]),  # Bottom right
            np.array([min_bounds[1], min_bounds[0], 0]),  # Top right
            np.array([max_bounds[1], min_bounds[0], 0]),  # Top left
        ]
        
        # Calculate center point in local coordinates
        center_local = np.mean(corners_local, axis=0)
        
        # Transform corners and center to global coordinates
        corners_global = []
        for corner in corners_local:
            # Transform to global space and offset by plane position
            global_pos = np.dot(local_to_global, corner) + normal * magnitude
            # Apply view rotation
            rotated_pos = np.dot(rotation_matrix, global_pos)
            # Project to screen space
            screen_pos = (int(rotated_pos[0] * scale + offset[0]),
                         int(-rotated_pos[2] * scale + offset[1]))
            corners_global.append(screen_pos)
        
        # Transform center point
        center_global = np.dot(local_to_global, center_local) + normal * magnitude
        center_rotated = np.dot(rotation_matrix, center_global)
        center_screen = (int(center_rotated[0] * scale + offset[0]),
                        int(-center_rotated[2] * scale + offset[1]))
        
        # Draw the sketch plane boundary
        pygame.draw.lines(screen, color, True, corners_global, 1)
        pygame.draw.circle(screen, color, center_screen, 3)

def render_edges(screen, vertices, edges, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    
    for edge in edges:
        v1, v2 = edge
        pygame.draw.line(screen, (0, 255, 255), projected_vertices[v1], projected_vertices[v2], 1)

def render_extrudes(screen, extrude_data, vertices, scale, offset, rotation_matrix, colors):
    """
    Render extrudes with their associated sketches
    """
    # Extract the extrudes list and sketch groups from the extrude_data dictionary
    extrudes = extrude_data['extrudes']
    sketch_groups = extrude_data['sketch_groups']
    
    # Create a mapping of sketch indices to their extrudes
    sketch_to_extrudes = defaultdict(list)
    
    # Populate the sketch_to_extrudes mapping using the sketch_groups structure
    for plane_key, group_info in sketch_groups.items():
        sketch_index = group_info['sketch_index']
        for extrude_index in group_info['extrudes']:
            if extrude_index < len(extrudes):  # Safety check
                sketch_to_extrudes[sketch_index].append((extrude_index, extrudes[extrude_index]))
    
    # Render extrudes organized by their sketch groups
    for sketch_index in sorted(sketch_to_extrudes.keys()):
        for extrude_index, extrude in sketch_to_extrudes[sketch_index]:
            # Use the extrude index to get the color
            sketch_color = colors[extrude_index % len(colors)]  # Add modulo to prevent index errors
            
            # Get the vertices for start and end points from matching pairs
            matching_pairs = extrude['matching_pairs']
            start_vertices = [vertices[idx1] for idx1, idx2 in matching_pairs]
            end_vertices = [vertices[idx2] for idx1, idx2 in matching_pairs]
            
            # Calculate centers
            start_center = np.mean(start_vertices, axis=0)
            end_center = np.mean(end_vertices, axis=0)
            
            # Apply rotation matrix
            rotated_start = np.dot(rotation_matrix, start_center)
            rotated_end = np.dot(rotation_matrix, end_center)
            
            # Project to screen space
            projected_start = (int(rotated_start[0] * scale + offset[0]), 
                             int(-rotated_start[2] * scale + offset[1]))
            projected_end = (int(rotated_end[0] * scale + offset[0]), 
                           int(-rotated_end[2] * scale + offset[1]))
            
            # Draw the extrude line and endpoints with the sketch's color
            pygame.draw.line(screen, sketch_color, projected_start, projected_end, 2)
            pygame.draw.circle(screen, sketch_color, projected_start, 4)
            pygame.draw.circle(screen, sketch_color, projected_end, 4)

def render_feature_tree(screen, feature_tree, colors, start_x=20, start_y=20):
    current_y = start_y
    line_height = 30
    font = pygame.font.SysFont(None, 24)


    for feature_index, feature in enumerate(feature_tree):
        color = colors[feature_index % len(colors)]
        
        # Render feature header
        sketch = feature['sketch']
        normal_str = sketch['normal'] if isinstance(sketch['normal'], str) else f"({sketch['normal'][0]:.2f}, {sketch['normal'][1]:.2f}, {sketch['normal'][2]:.2f})"
        feature_text = f"{feature['type']} sk{sketch['index'] + 1} @{normal_str} = {sketch['magnitude']:.3f}"
        text_surface = font.render(feature_text, True, color)
        screen.blit(text_surface, (start_x, current_y))
        current_y += line_height
        
        # Render associated extrudes
        for extrude_idx, extrude in enumerate(feature['extrudes']):
            extrude_text = (f"  > Extrude {extrude_idx + 1}: "
                            f"{extrude.get('start_magnitude', 0):.3f} > {extrude.get('end_magnitude', 0):.3f}")
            text_surface = font.render(extrude_text, True, color)
            screen.blit(text_surface, (start_x + 20, current_y))
            current_y += line_height
        
        current_y += 10
        
def render_sketches(screen, sketches, colors, start_y=500):
    try:
        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
    except:
        print("Warning: Font initialization failed")
        return
    
    sketch_width = 300
    sketch_height = 300
    padding = 20
    start_x = padding
    
    for sketch in sketches:
        # Calculate display position
        x = start_x + (sketch['index'] * (sketch_width + padding))
        y = start_y + padding
        
        # Create label
        normal_str = sketch['normal']
        if isinstance(normal_str, tuple):
            normal_str = f"({normal_str[0]:.2f}, {normal_str[1]:.2f}, {normal_str[2]:.2f})"
        label = f"sk{sketch['index'] + 1} {normal_str}={sketch['magnitude']:.3f}"
        text_surface = font.render(label, True, colors[sketch['index']])
        screen.blit(text_surface, (x, y - 20))
        
        # Transform edges to sketch plane coordinates
        normal = np.array(sketch['normal'])
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.array([1, 0, 0]) if abs(normal[1]) > 0.9 else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        # Transform matrix from global to local sketch space
        global_to_local = np.column_stack((x_axis, y_axis, z_axis)).T
        
        transformed_points = []
        for edge in sketch['edges']:
            start, end = edge
            # Convert to 3D points in sketch plane space
            start_3d = np.array([start[0], start[1], 0])
            end_3d = np.array([end[0], end[1], 0])
            transformed_points.extend([start_3d, end_3d])
            
        if not transformed_points:
            continue
            
        # Calculate bounds in sketch space
        points_array = np.array(transformed_points)
        min_bounds = np.min(points_array, axis=0)[:2]  # Only take x,y coordinates
        max_bounds = np.max(points_array, axis=0)[:2]
        
        # Calculate scale to fit in display area
        width = max_bounds[0] - min_bounds[0]
        height = max_bounds[1] - min_bounds[1]
        if width <= 0 or height <= 0:
            continue
            
        scale_x = (sketch_width - 2 * padding) / width
        scale_y = (sketch_height - 2 * padding) / height
        scale = min(scale_x, scale_y)
        
        # Calculate center of display area
        center_x = x + sketch_width / 2
        center_y = y + sketch_height / 2
        
        # Draw edges in sketch space
        for edge in sketch['edges']:
            start, end = edge
            # Convert to display coordinates
            start_display = (
                int(center_x + (start[0] - (min_bounds[0] + width/2)) * scale),
                int(center_y + (start[1] - (min_bounds[1] + height/2)) * scale)
            )
            end_display = (
                int(center_x + (end[0] - (min_bounds[0] + width/2)) * scale),
                int(center_y + (end[1] - (min_bounds[1] + height/2)) * scale)
            )
            pygame.draw.line(screen, colors[sketch['index']], start_display, end_display, 2)
