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


class DifferentialEvolution:
    def __init__(self, func, dimension, lower_bound, upper_bound, NP=20, F=0.5, CR=0.5, Gmax=50):
        self.func = func
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.NP = NP  # velikost populace
        self.F = F    # rozdilná hmotnost
        self.CR = CR  # pravděpodobnost křížení
        self.Gmax = Gmax  # počet generací
        self.population = [Solution(dimension, lower_bound, upper_bound) for _ in range(NP)]
        for solution in self.population:
            solution.f = self.func(solution.parameters)

    def evolve(self):
        history = []
        for g in range(self.Gmax):
            new_population = []
            for i, target in enumerate(self.population):
                # Mutace
                indices = np.random.choice([idx for idx in range(self.NP) if idx != i], 3, replace=False)
                r1, r2, r3 = [self.population[idx].parameters for idx in indices]
                mutant_vector = r1 + self.F * (r2 - r3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Křížení
                trial_vector = np.copy(target.parameters)
                j_rand = np.random.randint(self.dimension)
                for j in range(self.dimension):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial_vector[j] = mutant_vector[j]

                # Výběr
                trial_solution = Solution(self.dimension, self.lower_bound, self.upper_bound)
                trial_solution.parameters = trial_vector
                trial_solution.f = self.func(trial_vector)

                if trial_solution.f <= target.f:
                    new_population.append(trial_solution)
                else:
                    new_population.append(target)
            
            self.population = new_population
            best_solution = min(self.population, key=lambda sol: sol.f)
            history.append(best_solution.parameters.copy())

        return np.array(history)


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

    de = DifferentialEvolution(func, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound, NP=20, F=0.5, CR=0.5, Gmax=50)
    history_de = de.evolve()

    visualize(
        func,
        lower_bound,
        upper_bound,
        histories=[history_de],
        labels=["Differential Evolution"],
        colors=['r']
    )
