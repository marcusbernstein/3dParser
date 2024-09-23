import pygame
import math
import sys
from collections import defaultdict

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 750
FPS = 60
ROTATION_SPEED = 2  # Speed of rotation
AXIS_LENGTH = 100  # Length of the axes
SCALE = 6  # Default scaling factor

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Load the STL file
with open("test file", "r") as file:
    stl_text = file.read()

# Function to parse the STL text and return vertices and edges
def parse_stl(stl_text):
    vertices = []
    edges = []
    vertex_map = {}  # To map vertex coordinates to indices
    current_face = []

    lines = stl_text.splitlines()
    
    for line in lines:
        if line.strip().startswith("vertex"):
            parts = line.strip().split()
            vertex = tuple(map(float, parts[1:4]))

            if vertex not in vertex_map:
                vertex_map[vertex] = len(vertices)
                vertices.append(vertex)

            current_face.append(vertex_map[vertex])

        if line.strip().startswith("endfacet"):
            # Add edges for the current face
            for i in range(len(current_face)):
                start = current_face[i]
                end = current_face[(i + 1) % len(current_face)]
                edges.append((start, end))
            current_face = []

    return vertices, edges

# Matrix multiplication
def multiply_matrices(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

# Create rotation matrix for yaw, pitch, and roll
def create_rotation_matrix(yaw, pitch, roll):
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)

    yaw_matrix = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ]

    pitch_matrix = [
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ]

    roll_matrix = [
        [math.cos(roll), 0, math.sin(roll)],
        [0, 1, 0],
        [-math.sin(roll), 0, math.cos(roll)]
    ]

    rotation_matrix = multiply_matrices(yaw_matrix, pitch_matrix)
    rotation_matrix = multiply_matrices(rotation_matrix, roll_matrix)

    return rotation_matrix

# Apply rotation matrix
def apply_rotation_matrix(point, matrix):
    x, y, z = point
    x_new = x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0]
    y_new = x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1]
    z_new = x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2]
    return [x_new, y_new, z_new]

# Project 3D point onto 2D surface
def project(point, offset_x=0, offset_y=0):
    x, y, z = point
    x_proj = int(x * SCALE + WIDTH // 2 + offset_x)
    y_proj = int(-z * SCALE + HEIGHT // 2 + offset_y)
    return (x_proj, y_proj)

# Detect flat surfaces and group them by plane
def detect_flat_surfaces(vertices, edges):
    flat_surfaces = defaultdict(list)
    
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        # Check if the two vertices are on the same flat surface (z-coordinates are close)
        if abs(v1[2] - v2[2]) < 0.01:
            flat_surfaces[v1[2]].extend([edge[0], edge[1]])

    # Group into distinct surfaces
    surface_groups = []
    for key, surface in flat_surfaces.items():
        unique_vertices = list(set(surface))
        if len(unique_vertices) >= 4:  # We only care about surfaces with >= 4 vertices
            surface_groups.append(unique_vertices)

    return surface_groups

# Main loop
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("STL with Bounding Box Detection")
    clock = pygame.time.Clock()

    # Parse the STL vertices and edges
    stl_vertices, stl_edges = parse_stl(stl_text)

    # Detect flat surfaces
    flat_surfaces = detect_flat_surfaces(stl_vertices, stl_edges)

    # Initial rotation matrix
    rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation_step = create_rotation_matrix(-ROTATION_SPEED, 0, 0)
            rotation_matrix = multiply_matrices(rotation_matrix, rotation_step)
        if keys[pygame.K_RIGHT]:
            rotation_step = create_rotation_matrix(ROTATION_SPEED, 0, 0)
            rotation_matrix = multiply_matrices(rotation_matrix, rotation_step)
        if keys[pygame.K_UP]:
            rotation_step = create_rotation_matrix(0, -ROTATION_SPEED, 0)
            rotation_matrix = multiply_matrices(rotation_matrix, rotation_step)
        if keys[pygame.K_DOWN]:
            rotation_step = create_rotation_matrix(0, ROTATION_SPEED, 0)
            rotation_matrix = multiply_matrices(rotation_matrix, rotation_step)

        screen.fill((0, 0, 0))

        # Draw STL object
        projected_stl = [apply_rotation_matrix(v, rotation_matrix) for v in stl_vertices]
        for edge in stl_edges:
            pygame.draw.line(screen, WHITE, project(projected_stl[edge[0]]), project(projected_stl[edge[1]]), 2)

        # Draw bounding boxes around flat surfaces with >= 4 vertices
        for surface_group in flat_surfaces:
            face_points = [stl_vertices[i] for i in surface_group]
            if len(face_points) >= 4:
                x_vals = [v[0] for v in face_points]
                y_vals = [v[1] for v in face_points]
                z_vals = [v[2] for v in face_points]

                min_x, max_x = min(x_vals), max(x_vals)
                min_y, max_y = min(y_vals), max(y_vals)

                # Calculate bounding box with 1.5x margin
                width = (max_x - min_x) * 1.5
                height = (max_y - min_y) * 1.5

                # Center of the bounding box
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2

                # Project bounding box points
                top_left = project(apply_rotation_matrix([center_x - width / 2, center_y - height / 2, z_vals[0]], rotation_matrix))
                top_right = project(apply_rotation_matrix([center_x + width / 2, center_y - height / 2, z_vals[0]], rotation_matrix))
                bottom_left = project(apply_rotation_matrix([center_x - width / 2, center_y + height / 2, z_vals[0]], rotation_matrix))
                bottom_right = project(apply_rotation_matrix([center_x + width / 2, center_y + height / 2, z_vals[0]], rotation_matrix))

                # Draw the bounding box
                pygame.draw.line(screen, RED, top_left, top_right, 2)
                pygame.draw.line(screen, RED, top_right, bottom_right, 2)
                pygame.draw.line(screen, RED, bottom_right, bottom_left, 2)
                pygame.draw.line(screen, RED, bottom_left, top_left, 2)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
