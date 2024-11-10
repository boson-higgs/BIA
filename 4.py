import numpy as np
import matplotlib.pyplot as plt
import random

# Města a jejich vzdálenosti
cities = ["Praha", "Bratislava", "Berlín", "Budapešť", "Moskva", "Ankara"]

# Matice vzdáleností
distance_matrix = np.array([
    [0, 328, 349, 1929, 1822, 2412],   # Praha
    [328, 0, 677, 214, 1773, 1929],     # Bratislava
    [349, 677, 0, 886, 1812, 2012],     # Berlín
    [1929, 214, 886, 0, 1568, 1839],    # Budapešť
    [1822, 1773, 1812, 1568, 0, 1734],   # Moskva
    [2412, 1929, 2012, 1839, 1734, 0]    # Ankara
])

# Fixní souřadnice měst pro graf
city_positions = {
    "Praha": (0, 0),
    "Bratislava": (1, 1),
    "Berlín": (2, 0),
    "Budapešť": (1.5, 1.5),
    "Moskva": (4, 4),
    "Ankara": (3, 3)
}

# Parametry genetického algoritmu
population_size = 50
generations = 100
mutation_rate = 0.9

# Funkce pro výpočet celkové vzdálenosti cesty
def path_distance(path):
    return sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1)) + distance_matrix[path[-1], path[0]]

# Počáteční populace
population = [np.random.permutation(len(cities)) for _ in range(population_size)]

# Výběr rodičů (Turnajový výběr)
def select_parents(population):
    parent_a = min(random.sample(population, k=5), key=path_distance)
    parent_b = min(random.sample(population, k=5), key=path_distance)
    return parent_a, parent_b

# Křížení (Crossover)
def crossover(parent_a, parent_b):
    size = len(parent_a)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent_a[start:end]
    pos = end
    for city in parent_b:
        if city not in child:
            if pos >= size:
                pos = 0
            child[pos] = city
            pos += 1
    return child

# Mutace (náhodně prohodí dvě města)
def mutate(path):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        path[i], path[j] = path[j], path[i]
    return path

# Genetický algoritmus
for generation in range(generations):
    new_population = []
    for _ in range(population_size):
        parent_a, parent_b = select_parents(population)
        child = crossover(parent_a, parent_b)
        child = mutate(child)
        new_population.append(child)

    population = new_population
    best_path = min(population, key=path_distance)
    best_distance = path_distance(best_path)
    print(f"Generace {generation + 1}, Nejlepší vzdálenost: {best_distance:.2f} km")

    # Vizualizace nejlepší cesty s fixními pozicemi měst
    plt.clf()
    x = [city_positions[cities[city]][0] for city in best_path]
    y = [city_positions[cities[city]][1] for city in best_path]
    
    # Nakreslíme čáry pro každý úsek cesty
    for i in range(len(best_path) - 1):
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], 'o-')
    plt.plot([x[-1], x[0]], [y[-1], y[0]], 'o-')  # Propojíme poslední a první město

    # Přidáme názvy měst ke každému bodu na grafu
    for i, city_index in enumerate(best_path):
        city_name = cities[city_index]
        plt.text(x[i], y[i], city_name, fontsize=12, ha='right')

    plt.title(f"Generace {generation + 1}, Nejlepší vzdálenost: {best_distance:.2f} km")
    plt.pause(0.01)

plt.show()
