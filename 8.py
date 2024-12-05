import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry algoritmu
num_cities = 10
num_ants = 20
num_iterations = 100
alpha = 1.0  # vliv feromonů
beta = 2.0   # vliv viditelnosti
evaporation_rate = 0.5  # rychlost odpařování
Q = 100  # konstantní přenos feromonů

# Generování měst (náhodné souřadnice)
np.random.seed(42)
cities = np.random.rand(num_cities, 2) * 100
distance_matrix = np.linalg.norm(cities[:, None, :] - cities[None, :, :], axis=2)
visibility = 1 / (distance_matrix + np.eye(num_cities) * 1e10)

# Inicializace feromonové matice
pheromones = np.ones((num_cities, num_cities))

# Funkce pro výpočet pravděpodobnosti
def calculate_probabilities(current_city, unvisited, pheromones, visibility, alpha, beta):
    tau_eta = (pheromones[current_city, unvisited] ** alpha) * (visibility[current_city, unvisited] ** beta)
    return tau_eta / tau_eta.sum()

# Inicializace grafu
fig, ax = plt.subplots()
scat = ax.scatter(cities[:, 0], cities[:, 1], c="red", s=100, label="Cities")
lines = []
for _ in range(num_ants):
    line, = ax.plot([], [], 'b-', alpha=0.5)
    lines.append(line)

# Linie pro nejlepší cestu
best_line, = ax.plot([], [], 'g-', linewidth=2, label="Best Path")  # Zelená čára pro nejlepší trasu

ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Historie animace
paths = []
best_distance = float("inf")  # Nejlepší vzdálenost
best_path = None

# Hlavní smyčka ACO
def aco(iteration):
    global pheromones, best_distance, best_path
    new_paths = []
    new_distances = []

    for ant in range(num_ants):
        path = []
        current_city = np.random.randint(num_cities)
        unvisited = list(range(num_cities))
        unvisited.remove(current_city)
        path.append(current_city)

        # Budování cesty
        while unvisited:
            probabilities = calculate_probabilities(current_city, unvisited, pheromones, visibility, alpha, beta)
            next_city = np.random.choice(unvisited, p=probabilities)
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        path.append(path[0])  # návrat do výchozího bodu
        new_paths.append(path)
        distance = sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
        new_distances.append(distance)

        # Aktualizace nejlepší vzdálenosti
        if distance < best_distance:
            best_distance = distance
            best_path = path

    # Aktualizace feromonů
    pheromones *= (1 - evaporation_rate)
    for path, dist in zip(new_paths, new_distances):
        for i in range(len(path) - 1):
            pheromones[path[i], path[i + 1]] += Q / dist

    paths.append((new_paths, new_distances))

    # Aktualizace cest mravenců
    for ant, path in enumerate(new_paths):
        lines[ant].set_data(cities[path, 0], cities[path, 1])

    # Aktualizace nejlepší cesty
    if best_path is not None:
        best_line.set_data(cities[best_path, 0], cities[best_path, 1])

    ax.set_title(f"Ant Colony Optimization - TSP\nBest Distance: {best_distance:.2f}")
    
    print(f"Iteration {iteration + 1}: Best Distance = {best_distance:.2f}")

ani = FuncAnimation(fig, aco, frames=num_iterations, interval=200, repeat=False)

#ani.save("aco_tsp_animation.gif", writer="pillow")

plt.show()
