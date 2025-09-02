import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import heapq


class SimulationMetrics:
    """Clase para recopilar métricas básicas de la simulación"""

    def __init__(self):
        self.escapes = 0
        self.captures = 0
        self.episodes = 0

    def add_episode(self, steps, escaped):
        self.episodes += 1
        if escaped:
            self.escapes += 1
        else:
            self.captures += 1

    def get_escape_rate(self):
        return (self.escapes / self.episodes * 100) if self.episodes > 0 else 0.0

# ===============================
# Generar laberinto aleatorio
# ===============================
def generate_maze(rows, cols, extra_paths=0.15, seed=None):
    """Genera laberinto con semilla opcional para reproducibilidad"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
    # Añadir caminos extra para romper la unicidad
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if maze[r, c] == 1 and random.random() < extra_paths:
                maze[r, c] = 0

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


# ===============================
# Estrategia A*
# ===============================
def astar_path(maze, start, goal, chaser, safety_weight=0.3, memory=None):
    if memory is None:
        memory = set()

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruir camino incluyendo posicion actual
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]

        for dr, dc in moves:
            neighbor = (current[0] + dr, current[1] + dc)
            if not in_bounds(neighbor[0], neighbor[1], maze) or maze[neighbor] == 1:
                continue

            # Penalizar posiciones en memoria
            memory_penalty = 10 if neighbor in memory else 0
            # 1 es el costo base de un movimiento
            tentative_g = g_score[current] + 1 + memory_penalty

            h = improved_heuristic(neighbor, goal, chaser, safety_weight)
            f = tentative_g + h

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (f, neighbor))
    return []  # No hay ruta


def improved_heuristic(neighbor, goal, chaser, safety_weight):
    """
    Heurística admisible que nunca subestima el costo real
    """
    goal_distance = manhattan_dist(neighbor, goal)
    chaser_distance = manhattan_dist(neighbor, chaser)

    # Penalización suave por peligro que mantiene admisibilidad
    if chaser_distance <= 3:
        danger_penalty = (4 - chaser_distance) * (safety_weight * 0.3)
        return goal_distance + danger_penalty
    else:
        return goal_distance


def evader_strategy_astar(maze, evader, chaser, memory, **kwargs):
    """
    Estrategia A* que usa memoria para evitar ciclos y planifica rutas completas
    """
    goal = kwargs.get('goal', (maze.shape[0]-2, maze.shape[1]-2))
    safety_weight = kwargs.get('safety_weight', 0.3)

    # Convertir memory deque a set para búsqueda O(1)
    memory_set = set(memory) if memory else set()

    # Buscar camino completo usando A*
    path = astar_path(maze, evader, goal, chaser, safety_weight, memory_set)

    # Si encontró un camino válido, tomar el siguiente paso
    if len(path) > 1:
        next_move = path[1]
        # print(f"Siguiente movimiento: {evader} -> {next_move}")
        return next_move

    # Movimiento de emergencia si A* no encuentra camino
    emergency = emergency_move(maze, evader, chaser, memory_set)
    return emergency


def emergency_move(maze, evader, chaser, memory_set):
    """
    Movimiento de emergencia cuando A* no encuentra camino viable
    """
    best_move = evader
    max_distance = -1

    # Buscar el movimiento que más aleje del perseguidor
    for dr, dc in moves:
        nr, nc = evader[0] + dr, evader[1] + dc
        if in_bounds(nr, nc, maze) and maze[nr, nc] == 0:
            distance = manhattan_dist((nr, nc), chaser)

            # Preferir movimientos que no estén en memoria
            if (nr, nc) not in memory_set and distance > max_distance:
                max_distance = distance
                best_move = (nr, nc)
            # Si todos los movimientos están en memoria, tomar el que más aleje
            elif max_distance == -1 and distance > max_distance:
                max_distance = distance
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
def draw_maze(maze, evader, chaser, title='Simulacion', step=0):
    display = np.copy(maze)
    display[evader] = 2  # Evadidor
    display[chaser] = 3  # Perseguidor
    cmap = plt.cm.get_cmap("tab20", 4)
    plt.imshow(display, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"{title} - Paso {step}")
    plt.axis("off")
    plt.pause(0.1)

# ===============================
# Simulación
# ===============================
rows, cols = 21, 21
maze = generate_maze(rows, cols, extra_paths=0.15)

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
    print("posicion del evasor: ",evader_pos)

    # Chequear captura
    if chaser_pos == evader_pos:
        plt.clf()
        draw_maze(maze, evader_pos, chaser_pos)
        plt.title("¡Capturado!", fontsize=16)
        plt.ioff()
        plt.show()
        break
    if evader_pos == (rows - 2, cols - 2):
        plt.clf()
        draw_maze(maze, evader_pos, chaser_pos)
        plt.title("¡Escapó!", fontsize=16)
        plt.ioff()
        plt.show()
        break

    time.sleep(0.05)
