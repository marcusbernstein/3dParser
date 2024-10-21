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
        
def render_sketch_planes(screen, vertices, sketch_planes, scale, offset, rotation_matrix, padding=5):
    for axis, plane_magnitude, covered_vertices in sketch_planes:
        # Extract the vertices on this sketch plane
        plane_vertices = np.array([vertices[v] for v in covered_vertices])
        
        # Find the minimum and maximum coordinates for the other two axes (not on the plane axis)
        min_coords = np.min(plane_vertices, axis=0)
        max_coords = np.max(plane_vertices, axis=0)
        center = (min_coords + max_coords) / 2

        # Add padding
        min_coords -= padding
        max_coords += padding

        # Create a rectangle (4 corners) based on the axis of the sketch plane
        if axis == 'x':
            # Plane parallel to YZ, at x = plane_magnitude
            rectangle = [
                [plane_magnitude, min_coords[1], min_coords[2]],  # Bottom-left
                [plane_magnitude, max_coords[1], min_coords[2]],  # Top-left
                [plane_magnitude, max_coords[1], max_coords[2]],  # Top-right
                [plane_magnitude, min_coords[1], max_coords[2]],  # Bottom-right
                [plane_magnitude, center[0], center[1]] # center   # Bottom-right
            ]
        elif axis == 'y':
            # Plane parallel to XZ, at y = plane_magnitude
            rectangle = [
                [min_coords[0], plane_magnitude, min_coords[2]],  # Bottom-left
                [max_coords[0], plane_magnitude, min_coords[2]],  # Top-left
                [max_coords[0], plane_magnitude, max_coords[2]],  # Top-right
                [min_coords[0], plane_magnitude, max_coords[2]],  # Bottom-right
                [center[0], plane_magnitude, center[1]] # center   # Bottom-right
            ]
        elif axis == 'z':
            # Plane parallel to XY, at z = plane_magnitude
            rectangle = [
                [min_coords[0], min_coords[1], plane_magnitude],  # Bottom-left
                [max_coords[0], min_coords[1], plane_magnitude],  # Top-left
                [max_coords[0], max_coords[1], plane_magnitude],  # Top-right
                [min_coords[0], max_coords[1], plane_magnitude],  # Bottom-right
                [center[0], center[1], plane_magnitude] # center
            ]

        # Rotate the rectangle points
        rotated_rectangle = [np.dot(rotation_matrix, point) for point in rectangle]

        # Project the rotated 3D rectangle to 2D space
        projected_rectangle = [(int(point[0] * scale + offset[0]), int(-point[2] * scale + offset[1]))
                               for point in rotated_rectangle]

        # Draw the rectangle as a parallelogram (connect the points)
        pygame.draw.line(screen, (0, 255, 0), projected_rectangle[0], projected_rectangle[1], 2)  # Bottom edge
        pygame.draw.line(screen, (0, 255, 0), projected_rectangle[1], projected_rectangle[2], 2)  # Right edge
        pygame.draw.line(screen, (0, 255, 0), projected_rectangle[2], projected_rectangle[3], 2)  # Top edge
        pygame.draw.line(screen, (0, 255, 0), projected_rectangle[3], projected_rectangle[0], 2)  # Left edge     
        pygame.draw.circle(screen, (0, 255, 0), projected_rectangle[4], 3)

def render_extrudes(screen, extrude_data, vertices, scale, offset, rotation_matrix):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, extrude in enumerate(extrude_data['extrudes']):
        color = colors[i % len(colors)]

        # Calculate centerpoints
        start_vertices = [vertices[idx] for idx, _ in extrude['matching_vertices']]
        end_vertices = [vertices[idx] for _, idx in extrude['matching_vertices']]

        start_center = np.mean(start_vertices, axis=0)
        end_center = np.mean(end_vertices, axis=0)

        # Rotate and project points
        rotated_start = np.dot(rotation_matrix, start_center)
        rotated_end = np.dot(rotation_matrix, end_center)

        projected_start = (int(rotated_start[0] * scale + offset[0]), int(-rotated_start[2] * scale + offset[1]))
        projected_end = (int(rotated_end[0] * scale + offset[0]), int(-rotated_end[2] * scale + offset[1]))

        # Draw extrude line
        pygame.draw.line(screen, color, projected_start, projected_end, 2)

        # Draw points at start and end
        pygame.draw.circle(screen, color, projected_start, 4)
        pygame.draw.circle(screen, color, projected_end, 4)

def extrude_characterize(extrude_data: dict):
    """
    Render the detected extrudes and print debug information.

    Args:
        extrude_data (dict): Dictionary containing extrudes and debug information.
    """
    print("Debug Information:")
    for line in extrude_data['debug_info']:
        print(line)

    print("\nDetected Extrudes:")
    for i, extrude in enumerate(extrude_data['extrudes']):
        print(f"Extrude {i + 1}:")
        print(f"  Start: {extrude['start_plane']}")
        print(f"  End: {extrude['end_plane']}")
        print(f"  Direction: {extrude['direction']}")
        print(f"  Distance: {extrude['distance']}")
        print(f"  Vertices: {extrude['vertex_count']}")

    if not extrude_data['extrudes']:
        print("No extrudes detected.")

    # Placeholder for future rendering logic
    print("\nPlaceholder: Actual rendering of extrudes would happen here.")