import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import time

# ===============================
# Generar laberinto aleatorio
# ===============================
def generate_maze(rows, cols):
    maze = np.ones((rows, cols), dtype=int)  # 1 = pared
    start = (1, 1)
    maze[start] = 0

    stack = [start]
    while stack:
        r, c = stack[-1]
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        carved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and maze[nr, nc] == 1:
                maze[nr, nc] = 0
                maze[r + dr // 2, c + dc // 2] = 0
                stack.append((nr, nc))
                carved = True
                break
        if not carved:
            stack.pop()
    return maze

# ===============================
# Utilidades de movimiento
# ===============================
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def in_bounds(r, c, maze):
    return 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1]

def bfs_path(maze, start, goal):
    queue = deque([start])
    visited = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dr, dc in moves:
            nr, nc = current[0] + dr, current[1] + dc
            if in_bounds(nr, nc, maze) and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited[(nr, nc)] = current
                queue.append((nr, nc))
    path = []
    cur = goal
    while cur is not None and cur in visited:
        path.append(cur)
        cur = visited[cur]
    return path[::-1]

def manhattan_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
# ===============================
# Evasor mejorado
# ===============================

def evader_move(maze, evader, chaser, memory, lookahead=2):
    candidates = []
    max_score = -1

    for dr, dc in moves:
        nr, nc = evader[0] + dr, evader[1] + dc
        if in_bounds(nr, nc, maze) and maze[nr, nc] == 0 and (nr, nc) not in memory:
            # Puntuar basado en distancia futura después de lookahead pasos
            dist_now = manhattan_dist((nr, nc), chaser)
            future_dist = dist_now

            # Pequeña simulación para lookahead
            temp_pos = (nr, nc)
            for _ in range(lookahead - 1):
                temp_pos = _best_next_step(maze, temp_pos, chaser)
                future_dist = manhattan_dist(temp_pos, chaser)

            score = future_dist
            if score > max_score:
                candidates = [(nr, nc)]
                max_score = score
            elif score == max_score:
                candidates.append((nr, nc))

    if not candidates:
        # Si no hay candidatos válidos, moverse aunque sea a memoria
        for dr, dc in moves:
            nr, nc = evader[0] + dr, evader[1] + dc
            if in_bounds(nr, nc, maze) and maze[nr, nc] == 0:
                candidates.append((nr, nc))
    # Elegir aleatoriamente en caso de empate
    return random.choice(candidates) if candidates else evader

def _best_next_step(maze, pos, chaser):
    best_move = pos
    max_dist = -1
    for dr, dc in moves:
        nr, nc = pos[0] + dr, pos[1] + dc
        if in_bounds(nr, nc, maze) and maze[nr, nc] == 0:
            dist = manhattan_dist((nr, nc), chaser)
            if dist > max_dist:
                max_dist = dist
                best_move = (nr, nc)
    return best_move

#----evasor anterior
'''
def evader_move(maze, evader, chaser):
    best_move = evader
    max_dist = -1
    for dr, dc in moves:
        nr, nc = evader[0] + dr, evader[1] + dc
        if in_bounds(nr, nc, maze) and maze[nr, nc] == 0:
            dist = manhattan_dist((nr, nc), chaser)
            if dist > max_dist:
                max_dist = dist
                best_move = (nr, nc)
    return best_move
'''


def chaser_move(maze, chaser, evader):
    path = bfs_path(maze, chaser, evader)
    if len(path) > 1:
        return path[1]
    return chaser

# ===============================
# Visualización
# ===============================
def draw_maze(maze, evader, chaser):
    display = np.copy(maze)
    display[evader] = 2  # Evadidor
    display[chaser] = 3  # Perseguidor
    cmap = plt.cm.get_cmap("tab20", 4)
    plt.imshow(display, cmap=cmap, vmin=0, vmax=3)
    plt.axis("off")
    plt.pause(0.1)

# ===============================
# Simulación
# ===============================
rows, cols = 21, 21
maze = generate_maze(rows, cols)

# Posiciones iniciales
evader_pos = (1, 1)
chaser_pos = (rows - 2, cols - 2)
evader_memory = deque(maxlen=5)

plt.figure(figsize=(6, 6))
plt.ion()

for step in range(200):
    plt.clf()
    draw_maze(maze, evader_pos, chaser_pos)

    # Mover evasor y perseguidor
    evader_memory.append(evader_pos)
    #new_evader = evader_move(maze, evader_pos, chaser_pos)
    new_evader = evader_move(maze, evader_pos, chaser_pos, evader_memory, lookahead=2)
    new_chaser = chaser_move(maze, chaser_pos, new_evader)

    evader_pos, chaser_pos = new_evader, new_chaser

    # Chequear captura
    if chaser_pos == evader_pos:
        plt.clf()
        draw_maze(maze, evader_pos, chaser_pos)
        plt.title("¡Capturado!", fontsize=16)
        plt.ioff()
        plt.show()
        break

    time.sleep(0.05)
