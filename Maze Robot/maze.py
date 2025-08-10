import numpy as np
import matplotlib.pyplot as plt
import random

# Parámetros del mapa
rows, cols = 20, 20
prob_wall = 0.25  # Probabilidad de un muro

# Generar mapa inicial con más caminos libres
grid = np.where(np.random.rand(rows, cols) < prob_wall, 1, 0)

# Asegurar que haya caminos: aplicar un random walk desde el inicio
def random_walk(grid, steps=200):
    r, c = 0, 0
    for _ in range(steps):
        grid[r, c] = 0  # camino libre
        dr, dc = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        r = max(0, min(rows-1, r+dr))
        c = max(0, min(cols-1, c+dc))
    return grid

grid = random_walk(grid, steps=300)

# Definir inicio y meta
start = (0, 0)
goal = (rows-1, cols-1)
grid[start] = 2  # inicio
grid[goal] = 3   # meta

# Visualizar mapa con matplotlib
plt.figure(figsize=(6, 6))
cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
plt.imshow(grid, cmap=cmap, origin='upper')
plt.xticks([])
plt.yticks([])
plt.title("Mapa con más caminos y meta")
plt.show()
