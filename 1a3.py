######### siv0017

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension  # Počet rozměrů
        self.lB = lower_bound  # Dolní mez pro všechny parametry
        self.uB = upper_bound  # Horní mez pro všechny parametry
        self.parameters = np.random.uniform(self.lB, self.uB, self.dimension)  # Náhodná inicializace parametrů
        self.f = np.inf  # Inicializace hodnoty objektivní funkce jako nekonečna
    
    # Sphere funkce: f(x) = sum(x_i^2)
    def sphere(self, x):
        return np.sum(x ** 2)  # Minimální hodnota je 0 (když jsou všechny x_i = 0)
    
    # Ackley funkce: f(x) = -20 * exp(-0.2 * sqrt(mean(x^2))) - exp(mean(cos(2 * pi * x))) + 20 + e
    def ackley(self, x):
        return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
        # Globální minimum je na (0, 0) s hodnotou 0.
    
    # Rastrigin funkce: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    def rastrigin(self, x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
        # Minimální hodnota je 0 (když jsou všechny x_i = 0).

    # Rosenbrock funkce: f(x) = sum(100*(x_i+1 - x_i^2)^2 + (x_i - 1)^2)
    def rosenbrock(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        # Minimální hodnota je 0 (když jsou všechny x_i = 1).

    # Griewank funkce: f(x) = sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i))) + 1
    def griewank(self, x):
        sum_part = np.sum(x ** 2 / 4000)
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1
        # Minimální hodnota je 0 (když jsou všechny x_i = 0).

    # Schwefel funkce: f(x) = 418.9829*n - sum(x_i*sin(sqrt(abs(x_i))))
    def schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        # Minimální hodnota je 0 (když jsou všechny x_i = 420.9687).

    # Lévy funkce: f(x) = sin^2(pi*w_1) + sum((w_i - 1)^2*(1 + 10*sin^2(pi*w_i))) + (w_n - 1)^2*(1 + sin^2(2*pi*w_n))
    def levy(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3
        # Minimální hodnota je 0 (když jsou všechny x_i = 1).

    # Michalewicz funkce: f(x) = -sum(sin(x_i)*(sin(i*x_i^2/pi))^2*m)
    def michalewicz(self, x, m=10):
        return -np.sum(np.sin(x) * (np.sin(np.arange(1, len(x) + 1) * x ** 2 / np.pi)) ** (2 * m))
        # Minimální hodnota závisí na m, obvykle na hodnotě blízko 2.2.

    # Zakharov funkce: f(x) = sum(x_i^2) + (0.5*sum(i*x_i))^2 + (0.5*sum(i*x_i))^4
    def zakharov(self, x):
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2 ** 2 + sum2 ** 4
        # Minimální hodnota je 0 (když jsou všechny x_i = 0).


class BlindSearch:
    def __init__(self, func, dimension, lower_bound, upper_bound, iterations=1000):
        self.func = func  # Objektivní funkce, kterou se snažíme minimalizovat
        self.dimension = dimension  # Počet dimenzí
        self.lower_bound = lower_bound  # Dolní hranice pro všechny parametry
        self.upper_bound = upper_bound  # Horní hranice pro všechny parametry
        self.iterations = iterations  # Počet iterací hledání

    def search(self):
        history = []  # Historie nalezených řešení
        for _ in range(self.iterations):
            # Generování nového náhodného řešení
            solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
            history.append(solution)  # Uložení řešení do historie
        return np.array(history)  # Vrátí historii jako numpy pole
    


class SimulatedAnnealing:
    def __init__(self, func, dimension, lower_bound, upper_bound, iterations=1000, initial_temp=1000, cooling_rate=0.99):
        self.func = func  # Objektivní funkce, kterou se snažíme minimalizovat
        self.dimension = dimension  # Počet dimenzí
        self.lower_bound = lower_bound  # Dolní hranice pro všechny parametry
        self.upper_bound = upper_bound  # Horní hranice pro všechny parametry
        self.iterations = iterations  # Počet iterací hledání
        self.initial_temp = initial_temp  # Počáteční teplota pro simulované žíhání
        self.cooling_rate = cooling_rate  # Rychlost ochlazování

    def search(self):
        # Inicializace náhodného řešení
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        current_energy = self.func(current_solution)  # Výpočet hodnoty objektivní funkce pro aktuální řešení
        best_solution = current_solution.copy()  # Uchování nejlepšího nalezeného řešení
        best_energy = current_energy  # Uchování nejlepší hodnoty objektivní funkce
        history = [current_solution]  # Historie řešení

        temp = self.initial_temp  # Nastavení počáteční teploty

        for _ in range(self.iterations):
            # Generování nového sousedního řešení
            new_solution = current_solution + np.random.normal(0, 1, self.dimension)  # Perturbace aktuálního řešení
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)  # Udržení nového řešení v rámci hranic
            new_energy = self.func(new_solution)  # Výpočet hodnoty objektivní funkce pro nové řešení

            # Rozhodnutí, zda přijmout nové řešení
            if new_energy < current_energy or np.random.rand() < np.exp((current_energy - new_energy) / temp):
                current_solution = new_solution  # Aktualizace aktuálního řešení
                current_energy = new_energy  # Aktualizace hodnoty objektivní funkce

                # Aktualizace nejlepšího řešení
                if new_energy < best_energy:
                    best_solution = new_solution  # Aktualizace nejlepšího řešení
                    best_energy = new_energy  # Aktualizace nejlepší hodnoty

            # Snižování teploty
            temp *= self.cooling_rate  # Snížení teploty
            history.append(current_solution)  # Uložení aktuálního řešení do historie

        return np.array(history)  # Vrátí historii jako numpy pole



# Vizualizace
def visualize(func, lower_bound, upper_bound, histories, labels, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Vytvoření meshgrid pro povrch
    X = np.linspace(lower_bound, upper_bound, 100)
    Y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([func(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)

    # Vykreslení povrchu
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # Vykreslení historie pro každý algoritmus
    for history, label, color in zip(histories, labels, colors):
        history_2d = history[:, :2]
        Z_history = np.array([func(np.array([x, y])) for x, y in history_2d])
        ax.scatter(history_2d[:, 0], history_2d[:, 1], Z_history, color=color, label=label)

    ax.set_title(f'{func.__name__}')
    ax.legend()
    plt.show()

# Horní a dolní mez
function_bounds = {
    'Sphere': (-5, 5),
    'Ackley': (-5, 5),
    'Rastrigin': (-5.12, 5.12),
    'Rosenbrock': (-2, 2),
    'Griewank': (-600, 600),
    'Schwefel': (-500, 500),
    'Levy': (-10, 10),
    'Michalewicz': (0, np.pi),
    'Zakharov': (-10, 10)
}


dimension = 2
iterations = 1000

labels = list(function_bounds.keys())
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'pink']

for label in labels:
    lower_bound, upper_bound = function_bounds[label]
    solution = Solution(dimension, lower_bound, upper_bound)
    
    func = getattr(solution, label.lower())
    
    blind_search = BlindSearch(func, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound, iterations=iterations)
    history_bs = blind_search.search()
    
    sim_anneal = SimulatedAnnealing(func, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound, iterations=iterations)
    history_sa = sim_anneal.search()
    
    visualize(
        func,
        lower_bound,
        upper_bound,
        histories=[history_bs, history_sa],
        labels=[f"Blind Search", f"Simulated Annealing"],
        colors=['blue', 'red']
    )