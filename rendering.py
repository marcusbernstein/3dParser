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

def render_sketch_planes(screen, vertices, sketches, scale, offset, rotation_matrix, colors, padding=5):
    for sketch in sketches:
        color = colors[sketch['index']]
        axis = sketch['axis']
        plane_magnitude = sketch['magnitude']
        
        # Convert 2D edges back to 3D vertices for bounding box calculation
        plane_points = []
        for edge in sketch['edges']:
            start, end = edge
            # Convert 2D coordinates back to 3D based on the plane's axis
            if axis == 'x':
                plane_points.append([plane_magnitude, start[0], start[1]])
                plane_points.append([plane_magnitude, end[0], end[1]])
            elif axis == 'y':
                plane_points.append([start[0], plane_magnitude, start[1]])
                plane_points.append([end[0], plane_magnitude, end[1]])
            else:  # z
                plane_points.append([start[0], start[1], plane_magnitude])
                plane_points.append([end[0], end[1], plane_magnitude])
        
        if plane_points:
            # Convert to numpy array for easier calculations
            plane_points = np.array(plane_points)
            min_coords = np.min(plane_points, axis=0)
            max_coords = np.max(plane_points, axis=0)
            center = (min_coords + max_coords) / 2

            # Add padding to the bounding box
            min_coords -= padding
            max_coords += padding

            # Create the corners of the bounding rectangle based on the plane's axis
            if axis == 'x':
                rectangle = [
                    [plane_magnitude, min_coords[1], min_coords[2]],  
                    [plane_magnitude, max_coords[1], min_coords[2]],  
                    [plane_magnitude, max_coords[1], max_coords[2]],  
                    [plane_magnitude, min_coords[1], max_coords[2]],  
                    [plane_magnitude, center[1], center[2]]  # Center point
                ]
            elif axis == 'y':
                rectangle = [
                    [min_coords[0], plane_magnitude, min_coords[2]],  
                    [max_coords[0], plane_magnitude, min_coords[2]],  
                    [max_coords[0], plane_magnitude, max_coords[2]],  
                    [min_coords[0], plane_magnitude, max_coords[2]],  
                    [center[0], plane_magnitude, center[2]]  # Center point
                ]
            else:  # axis == 'z'
                rectangle = [
                    [min_coords[0], min_coords[1], plane_magnitude],  
                    [max_coords[0], min_coords[1], plane_magnitude],  
                    [max_coords[0], max_coords[1], plane_magnitude],  
                    [min_coords[0], max_coords[1], plane_magnitude],  
                    [center[0], center[1], plane_magnitude]  # Center point
                ]

            # Project the rectangle to screen space
            rotated_rectangle = [np.dot(rotation_matrix, point) for point in rectangle]
            projected_rectangle = [(int(point[0] * scale + offset[0]), 
                                  int(-point[2] * scale + offset[1]))
                                 for point in rotated_rectangle]

            # Draw the bounding box
            pygame.draw.line(screen, color, projected_rectangle[0], projected_rectangle[1], 1)
            pygame.draw.line(screen, color, projected_rectangle[1], projected_rectangle[2], 1)
            pygame.draw.line(screen, color, projected_rectangle[2], projected_rectangle[3], 1)
            pygame.draw.line(screen, color, projected_rectangle[3], projected_rectangle[0], 1)
            pygame.draw.circle(screen, color, projected_rectangle[4], 3)  # Center point

def render_sketch_contours(screen, vertices, sketch_plane_edges, extrude_data, scale, offset, rotation_matrix, colors):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]

    for plane in sketch_plane_edges:
        axis, magnitude, edges = plane
        plane_key = (axis, round(magnitude, 3))
        
        # Find the corresponding sketch index for this plane
        for start_plane, info in extrude_data['sketch_groups'].items():
            if start_plane == plane_key:
                color = colors[info['sketch_index']]
                for edge in edges:
                    v1, v2 = edge
                    if 0 <= v1 < len(projected_vertices) and 0 <= v2 < len(projected_vertices):
                        pygame.draw.line(screen, color, projected_vertices[v1], projected_vertices[v2], 3)
                break

def render_edges(screen, vertices, edges, scale, offset, rotation_matrix):
    rotated_vertices = [np.dot(rotation_matrix, vertex) for vertex in vertices]
    projected_vertices = [(int(vertex[0] * scale + offset[0]), int(-vertex[2] * scale + offset[1]))
                          for vertex in rotated_vertices]
    
    for edge in edges:
        v1, v2 = edge
        pygame.draw.line(screen, (0, 255, 255), projected_vertices[v1], projected_vertices[v2], 1)
        
def render_extrudes(screen, extrude_data, vertices, scale, offset, rotation_matrix, colors):
    # Create a mapping of sketch indices to their extrudes
    sketch_to_extrudes = {}
    for i, extrude in enumerate(extrude_data['extrudes']):
        start_plane = (extrude['start_plane'][0], round(extrude['start_plane'][1], 3))
        # Find which sketch this extrude belongs to
        for group_plane, info in extrude_data['sketch_groups'].items():
            if group_plane == start_plane:
                sketch_index = info['sketch_index']
                if sketch_index not in sketch_to_extrudes:
                    sketch_to_extrudes[sketch_index] = []
                sketch_to_extrudes[sketch_index].append((i, extrude))
                break
    
    # Render extrudes organized by their sketch groups
    for sketch_index in sorted(sketch_to_extrudes.keys()):
        sketch_color = colors[sketch_index]  # Use the sketch's color for all its extrudes
        
        for extrude_index, extrude in sketch_to_extrudes[sketch_index]:
            # Get the vertices for start and end points
            start_vertices = [vertices[idx] for idx, _ in extrude['matching_vertices']]
            end_vertices = [vertices[idx] for _, idx in extrude['matching_vertices']]

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

def render_feature_tree(screen, extrude_data, sketches, colors, start_x=20, start_y=20):
    try:
        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
    except:
        print("Warning: Font initialization failed")
        return

    current_y = start_y
    line_height = 30
    
    # Create a mapping of sketch indices to their extrudes
    sketch_to_extrudes = {}
    for i, extrude in enumerate(extrude_data['extrudes']):
        start_plane = (extrude['start_plane'][0], round(extrude['start_plane'][1], 3))
        # Find which sketch this extrude belongs to
        for group_plane, info in extrude_data['sketch_groups'].items():
            if group_plane == start_plane:
                sketch_index = info['sketch_index']
                if sketch_index not in sketch_to_extrudes:
                    sketch_to_extrudes[sketch_index] = []
                sketch_to_extrudes[sketch_index].append((i, extrude))
                break
    
    # Sort sketches by their index to maintain consistent ordering
    sorted_sketches = sorted(sketches, key=lambda x: x['index'])
    
    # Render each sketch and its dependent features
    for sketch in sorted_sketches:
        sketch_index = sketch['index']
        color = colors[sketch_index]
        
        # Draw sketch header
        sketch_text = f"sk{sketch_index + 1} ({sketch['axis']}={sketch['magnitude']:.3f})"
        text_surface = font.render(sketch_text, True, color)
        screen.blit(text_surface, (start_x, current_y))
        current_y += line_height
        
        # Draw all extrudes that depend on this sketch
        if sketch_index in sketch_to_extrudes:
            for extrude_index, extrude in sketch_to_extrudes[sketch_index]:
                direction = "+" if extrude['direction'] > 0 else "-"
                end_plane = f"{extrude['end_plane'][0]}={extrude['end_plane'][1]:.3f}"
                
                # Render extrude details under the sketch
                extrude_text = f"  > Extrude {extrude_index + 1}: {extrude['end_plane'][0]}{direction}{extrude['distance']}"
                text_surface = font.render(extrude_text, True, color)
                screen.blit(text_surface, (start_x + 20, current_y))
                current_y += line_height
        
        # Add a small gap between sketch groups
        current_y += 10

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

def render_sketches(screen, sketches, colors, start_y=500):
    try:
        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
    except:
        print("Warning: Font initialization failed")
        return
    
    sketch_width = 200  # Width allocated for each sketch
    sketch_height = 200  # Height allocated for each sketch
    padding = 20  # Padding between sketches
    start_x = padding  # Starting X position
    
    for sketch in sketches:
        # Calculate the position for this sketch
        x = start_x + (sketch['index'] * (sketch_width + padding))
        y = start_y + padding
        
        # Draw the sketch label
        label = f"sk{sketch['index'] + 1} ({sketch['axis']}={sketch['magnitude']:.3f})"
        text_surface = font.render(label, True, colors[sketch['index']])
        screen.blit(text_surface, (x, y - 20))
        
        # Find bounds of the sketch
        all_points = []
        for edge in sketch['edges']:
            all_points.extend([edge[0], edge[1]])
        
        if all_points:
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            # Calculate scale to fit in the allocated space
            width = max_x - min_x
            height = max_y - min_y
            if width > 0 and height > 0:
                scale_x = (sketch_width - 2 * padding) / width
                scale_y = (sketch_height - 2 * padding) / height
                scale = min(scale_x, scale_y)
                
                # Calculate center offset
                center_x = x + sketch_width / 2
                center_y = y + sketch_height / 2
                
                # Draw edges
                for edge in sketch['edges']:
                    start_point = (
                        int(center_x + (edge[0][0] - (min_x + width/2)) * scale),
                        int(center_y + (edge[0][1] - (min_y + height/2)) * scale)
                    )
                    end_point = (
                        int(center_x + (edge[1][0] - (min_x + width/2)) * scale),
                        int(center_y + (edge[1][1] - (min_y + height/2)) * scale)
                    )
                    pygame.draw.line(screen, colors[sketch['index']], start_point, end_point, 2)