import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Třída s testovacími funkcemi
class Solution:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension  # Počet rozměrů
        self.lB = lower_bound  # Dolní mez
        self.uB = upper_bound  # Horní mez
        self.parameters = np.random.uniform(self.lB, self.uB, self.dimension)  # Náhodná inicializace parametrů
        self.f = np.inf  # Inicializace hodnoty objektivní funkce jako nekonečna
    
    # Sphere funkce
    def sphere(self, x):
        return np.sum(x ** 2)
    
    # Ackley funkce
    def ackley(self, x):
        return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
    
    # Rastrigin funkce
    def rastrigin(self, x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    # Rosenbrock funkce
    def rosenbrock(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
    
    # Griewank funkce
    def griewank(self, x):
        sum_part = np.sum(x ** 2 / 4000)
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1
    
    # Schwefel funkce
    def schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    # Lévy funkce
    def levy(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3
    
    # Zakharov funkce
    def zakharov(self, x):
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2 ** 2 + sum2 ** 4

# Firefly Algoritmus
class FireflyAlgorithm:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=50, iterations=100, alpha=0.5, beta=1.0, gamma=1.0):
        self.func = func
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.iterations = iterations
        self.alpha = alpha  # Náhodná složka
        self.beta = beta  # Přitažlivost
        self.gamma = gamma  # Tlumení přitažlivosti

    def search(self):
        fireflies = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        fitness = np.array([self.func(firefly) for firefly in fireflies])
        history = []

        for _ in range(self.iterations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # Pohyb směrem k lepší světlušce
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta_effective = self.beta * np.exp(-self.gamma * r ** 2)
                        fireflies[i] += beta_effective * (fireflies[j] - fireflies[i]) + self.alpha * np.random.uniform(-1, 1, self.dimension)
                        fireflies[i] = np.clip(fireflies[i], self.lower_bound, self.upper_bound)
                        fitness[i] = self.func(fireflies[i])

            history.append(fireflies.copy())

        best_index = np.argmin(fitness)
        best_solution = fireflies[best_index]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness, np.array(history)

# Vizualizace animace
def animate(func, lower_bound, upper_bound, history):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Povrch funkce
    X = np.linspace(lower_bound, upper_bound, 100)
    Y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.array([func(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())])
    Z = Z.reshape(X.shape)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    scatter = ax.scatter([], [], [], color='red', label='Fireflies')

    def update(frame):
        positions = history[frame]
        Z_positions = np.array([func(pos) for pos in positions])
        scatter._offsets3d = (positions[:, 0], positions[:, 1], Z_positions)
        ax.set_title(f"Firefly Algorithm - Iteration {frame + 1}")

    ani = FuncAnimation(fig, update, frames=len(history), interval=200, repeat=False)
    plt.legend()
    plt.show()

# Parametry
function_bounds = {
    'Sphere': (-5, 5),
    'Ackley': (-5, 5),
    'Rastrigin': (-5.12, 5.12),
    'Rosenbrock': (-2, 2),
    'Griewank': (-600, 600),
    'Schwefel': (-500, 500),
    'Levy': (-10, 10),
    'Zakharov': (-10, 10)
}

dimension = 2
iterations = 50

# Spuštění a vizualizace
for label, (lower_bound, upper_bound) in function_bounds.items():
    solution = Solution(dimension, lower_bound, upper_bound)
    func = getattr(solution, label.lower())
    
    firefly = FireflyAlgorithm(func, dimension=dimension, lower_bound=lower_bound, upper_bound=upper_bound, iterations=iterations, population_size=20)
    best_solution, best_fitness, history_fa = firefly.search()
    
    print(f"{label}: Best solution: {best_solution}, Fitness: {best_fitness}")
    
    animate(func, lower_bound, upper_bound, history_fa)
