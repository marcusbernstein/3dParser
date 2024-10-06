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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    
    scale = 4
    offset = (WIDTH // 2, HEIGHT // 2)
    rotation_angle = 0.01
    
    rotation_matrix = np.eye(3)  # Identity matrix
    rotation_x = 0
    rotation_y = 0
    rotation_z = 0


    vertices, triangles = parse('tester.stl')
    surfaces = detect_flat_surfaces(vertices, triangles)
    print(len(surfaces))
    shapes = recognize_shapes(surfaces, vertices)
    #identified_3d_shapes = track_extrudes(shapes)

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

        rotation_matrix_x = [[1, 0, 0],
                         [0, math.cos(rotation_x), -math.sin(rotation_x)],
                         [0, math.sin(rotation_x), math.cos(rotation_x)]]
        rotation_matrix_y = [[math.cos(rotation_y), 0, math.sin(rotation_y)],
                         [0, 1, 0],
                         [-math.sin(rotation_y), 0, math.cos(rotation_y)]]
        rotation_matrix_z = [[math.cos(rotation_z), -math.sin(rotation_z), 0],
                         [math.sin(rotation_z), math.cos(rotation_z), 0],
                         [0, 0, 1]]

        rotation_matrix = np.dot(np.dot(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
        screen.fill(BLACK)
        
        render_stl(screen, vertices, triangles, scale, offset, rotation_matrix)
        render_shapes(screen, vertices, triangles, shapes, scale, offset, rotation_matrix, surfaces)

        pygame.display.flip()
        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    main()