import pygame
import math
from detection import *
from constants import *
from stl_parser import *
from rendering import *

def rotate(vertex, axis, angle):
    x, y, z = vertex

    if axis == 'x':
        return (x, 
                y * math.cos(angle) - z * math.sin(angle), 
                y * math.sin(angle) + z * math.cos(angle))
    elif axis == 'y':
        return (x * math.cos(angle) + z * math.sin(angle), 
                y, 
                -x * math.sin(angle) + z * math.cos(angle))
    elif axis == 'z':
        return (x * math.cos(angle) - y * math.sin(angle), 
                x * math.sin(angle) + y * math.cos(angle), 
                z)
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")

def calculate_rotation_matrix(angle_x, angle_y, angle_z):
    rotation_matrix_x = [[1, 0, 0],
                         [0, math.cos(angle_x), -math.sin(angle_x)],
                         [0, math.sin(angle_x), math.cos(angle_x)]]
    rotation_matrix_y = [[math.cos(angle_y), 0, math.sin(angle_y)],
                         [0, 1, 0],
                         [-math.sin(angle_y), 0, math.cos(angle_y)]]
    rotation_matrix_z = [[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z), math.cos(angle_z), 0],
                         [0, 0, 1]]
    return np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    scale = 7
    offset = (WIDTH // 2, HEIGHT // 2)
    rotation_angle = 0.01
    
    rotation_x = 0
    rotation_y = 0
    rotation_z = 0

    vertices, triangles, edges = parse('testersquare.stl')
    surfaces = detect_flat_surfaces(vertices, triangles)
    recognized_contours, recognized_circles = recognize_shapes(vertices, edges, surfaces)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation_z -= rotation_angle
        if keys[pygame.K_RIGHT]:
            rotation_z += rotation_angle
        if keys[pygame.K_UP]:
            rotation_x -= rotation_angle
        if keys[pygame.K_DOWN]:
            rotation_x += rotation_angle

        rotation_matrix = calculate_rotation_matrix(rotation_x, rotation_y, rotation_z)
        screen.fill(BLACK)
        
        render_stl(screen, vertices, triangles, scale, offset, rotation_matrix)
        render_shapes(screen, vertices, surfaces, recognized_contours, recognized_circles, scale, offset, rotation_matrix)

        pygame.display.flip()
        pygame.time.Clock().tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    main()