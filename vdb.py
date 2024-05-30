import pygame
import sys
import numpy as np
import copy

# Initialize Pygame
pygame.init()

# Set up the drawing window
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vertex Block Decent Demo")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Define font for displaying coordinates
font = pygame.font.Font(None, 36)

# Create a NumPy array of points (ensure it's integer type for pygame compatibility)
points_array = np.zeros((5,2),dtype=np.float32)

N = 5
rod_pos = np.zeros((N,2), dtype=np.float32)
rod_old_pos = np.zeros((N,2), dtype=np.float32)
rod_inertia = np.zeros((N,2), dtype=np.float32)
rod_vel = np.zeros((N,2), dtype=np.float32)
rod_mass = np.zeros(N, dtype=np.float32)
gravity = np.array([0, -10], dtype=np.float32)
vertAdjacentEdges = np.zeros((N,2), dtype=np.int32)
restLen = np.zeros(N-1, dtype=np.float32)
edges = np.zeros((N-1,2),dtype=np.int32)
h = 0.01666666
stiffness = 1.0e8

for i in range(N):
    rod_pos[i][0] = i * 0.15 + 0.5
    rod_pos[i][1] = 0.9
    rod_mass[i] = 1.0
    if i == N-1:
        vertAdjacentEdges[i] = [i-1, -1]
    else:
        vertAdjacentEdges[i] = [i-1, i]
rod_mass[-1] = 1000

for i in range(N-1):
    edges[i] = [i, i+1]
    restLen[i] = np.linalg.norm(rod_pos[i+1] - rod_pos[i])

def map_points():
    for i in range(N):
        points_array[i][0] = rod_pos[i][0] * WIDTH
        points_array[i][1] = HEIGHT - rod_pos[i][1] * HEIGHT

def simi_euler(h, rod_pos):
    rod_old_pos = copy.deepcopy(rod_pos)
    for i in range(N):
        if i == 0:
            rod_vel[i] = [0.0, 0.0]
        else:
            rod_vel[i] += gravity * h
        rod_inertia[i] = rod_pos[i] + rod_vel[i] * h
    rod_pos = rod_inertia
    return rod_old_pos


def update_vel(h):
    for i in range(N):
        rod_vel[i] = (rod_pos[i] - rod_old_pos[i]) / h


def solve(h):
    dtSqrReciprocal = 1.0 / (h**2)
    for i in range(1,N):
        hessian = np.zeros((2,2), dtype=np.float32)
        f = np.zeros((2,1), dtype=np.float32)
        
        f = rod_mass[i] * (rod_inertia[i] - rod_pos[i]) * dtSqrReciprocal
        hessian += rod_mass[i] * dtSqrReciprocal * np.eye(2, 2)
        
        for j in range(2):
            edge1 = vertAdjacentEdges[i][j]
            if edge1 != -1:
                vidx1, vidx2 = edges[edge1]
                
                diff = rod_pos[vidx1] - rod_pos[vidx2]
                l = np.linalg.norm(diff)
                l0 = restLen[edge1]
                H = stiffness * (np.eye(2,2) - (l0/l) * (np.eye(2,2) - np.outer(diff,diff)/(l**2)))
                hessian += H
                if vidx1 == i:
                    f += stiffness * (l0 - l) / l * diff
                else:
                    f -= stiffness * (l0 - l) / l * diff
        dx = np.linalg.solve(hessian, f)
        rod_pos[i] += dx                


# Run until the user asks to quit
running = True
substeps = 5
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill(WHITE)

    for i in range(substeps):
        rod_old_pos = simi_euler(h/substeps, rod_pos)
        for i in range(100):
            solve(h/substeps)
        update_vel(h/substeps)

    map_points()
    # Convert NumPy array to a list of tuples (pairs)
    points = [tuple(point) for point in points_array]
    # Draw the multi-segment line
    pygame.draw.lines(screen, BLACK, False, points, 5)

    # Draw the points and their coordinates
    for point in points:
        pygame.draw.circle(screen, RED, point, 5)
        text = font.render(f"({int(point[0])}, {int(point[1])})", True, BLACK)
        screen.blit(text, (point[0] + 10, point[1] - 20))

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
sys.exit()
