##### siv0017

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

# Parametry genetického algoritmu
population_size = 100   # Velikost populace
generations = 100       # Počet generací
mutation_rate = 200    # Pravděpodobnost mutace

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
    
    # Vezmeme úsek z rodiče A
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent_a[start:end]
    
    # Doplníme zbývající část z rodiče B
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

    # Vizualizace nejlepší cesty
    plt.clf()
    x = np.arange(len(cities))  # Osa X pro města
    y = best_path  # Osa Y pro pořadí měst podle nejlepší nalezené cesty

    # Nakreslíme čáry pro každý úsek cesty
    for i in range(len(best_path) - 1):
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], 'o-')
    # Propojíme poslední a první město, abychom uzavřeli smyčku
    plt.plot([x[-1], x[0]], [y[-1], y[0]], 'o-')
    
    # Přidáme názvy měst ke každému bodu na grafu
    for i, city_index in enumerate(best_path):
        plt.text(x[i], y[i], cities[city_index], fontsize=12, ha='right')
    
    plt.title(f"Generace {generation + 1}, Nejlepší vzdálenost: {best_distance:.2f} km")
    plt.pause(0.01)

plt.show()
