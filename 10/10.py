# siv0017

import numpy as np
import pandas as pd

# Algoritmus Teaching-Learning Based Optimization (TLBO)
# ---------------------------------------------
# Tato třída implementuje algoritmus TLBO pro optimalizaci. Algoritmus simuluje proces výuky a učení ve třídě, aby našel globální minimum dané cílové funkce.
class TLBO:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=30, max_evaluations=3000):
        # Cílová funkce k minimalizaci
        self.func = func
        # Dimenze prostoru řešení
        self.dimension = dimension
        # Dolní a horní hranice proměnných
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Velikost populace (počet studentů ve třídě)
        self.population_size = population_size
        # Maximální počet vyhodnocení cílové funkce
        self.max_evaluations = max_evaluations

        # Inicializace populace náhodně v rámci hranic
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        # Výpočet počátečních hodnot vhodnosti (fitness) pro populaci
        self.fitness = np.array([self.func(individual) for individual in self.population])

    def optimize(self):
        # Počet vyhodnocení cílové funkce
        evaluations = len(self.fitness)
        # Inicializace nejlepšího řešení a jeho hodnoty fitness
        best_solution = self.population[np.argmin(self.fitness)]
        best_fitness = np.min(self.fitness)

        while evaluations < self.max_evaluations:
            # Fáze výuky: studenti se učí od nejlepšího učitele
            mean_vector = np.mean(self.population, axis=0)  # Výpočet průměru populace
            teacher = self.population[np.argmin(self.fitness)]  # Identifikace nejlepšího učitele
            tf = np.random.randint(1, 3)  # Výukový faktor (1 nebo 2)

            # Aktualizace populace na základě znalostí učitele
            new_population = self.population + np.random.random((self.population_size, self.dimension)) * (teacher - tf * mean_vector)
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)  # Zajištění, že řešení zůstanou v rámci hranic

            # Vyhodnocení fitness nové populace
            new_fitness = np.array([self.func(individual) for individual in new_population])
            evaluations += self.population_size

            # Nahrazení starých řešení zlepšenými řešeními
            improvement = new_fitness < self.fitness
            self.population[improvement] = new_population[improvement]
            self.fitness[improvement] = new_fitness[improvement]

            # Fáze učení: studenti se učí od sebe navzájem
            for i in range(self.population_size):
                partner = np.random.choice([j for j in range(self.population_size) if j != i])  # Náhodný výběr partnera k učení
                if self.fitness[i] < self.fitness[partner]:  # Pokud je aktuální student lepší
                    new_solution = self.population[i] + np.random.random(self.dimension) * (self.population[i] - self.population[partner])
                else:  # Pokud je lepší partner
                    new_solution = self.population[i] + np.random.random(self.dimension) * (self.population[partner] - self.population[i])

                # Zajištění, že nové řešení je v rámci hranic
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                new_fitness = self.func(new_solution)
                evaluations += 1

                # Nahrazení aktuálního řešení, pokud je nové lepší
                if new_fitness < self.fitness[i]:
                    self.population[i] = new_solution
                    self.fitness[i] = new_fitness

            # Aktualizace nejlepšího řešení nalezeného dosud
            current_best = np.min(self.fitness)
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = self.population[np.argmin(self.fitness)]

        # Návrat nejlepšího řešení a jeho hodnoty fitness
        return best_solution, best_fitness

# ---------------------------------------------
# Cílové funkce
# ---------------------------------------------
# Kolekce standardních testovacích funkcí používaných pro optimalizaci. Tyto funkce se běžně používají k hodnocení výkonnosti optimalizačních algoritmů.
def sphere(x):
    """Funkce Sphere: Součet druhých mocnin vektoru. Globální minimum je 0 při x = [0, 0, ..., 0]."""
    return np.sum(x**2)

def ackley(x):
    """Funkce Ackley: Multimodální funkce s globálním minimem 0 při x = [0, 0, ..., 0]."""
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e

def rastrigin(x):
    """Funkce Rastrigin: Multimodální funkce s mnoha lokálními minimy. Globální minimum je 0 při x = [0, 0, ..., 0]."""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rosenbrock(x):
    """Funkce Rosenbrock: Nekonvexní funkce. Globální minimum je 0 při x = [1, 1, ..., 1]."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def griewank(x):
    """Funkce Griewank: Funkce s mnoha lokálními minimy. Globální minimum je 0 při x = [0, 0, ..., 0]."""
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def schwefel(x):
    """Funkce Schwefel: Funkce s mnoha lokálními minimy. Globální minimum je 0 při x = [420.9687, ..., 420.9687]."""
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x):
    """Funkce Levy: Multimodální funkce s globálním minimem při x = [1, 1, ..., 1]."""
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def michalewicz(x, m=10):
    """Funkce Michalewicz: Komplexní multimodální funkce. Globální minimum závisí na parametru m."""
    return -np.sum(np.sin(x) * (np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi))**(2 * m))

def zakharov(x):
    """Funkce Zakharov: Jednoduchá testovací funkce. Globální minimum je 0 při x = [0, 0, ..., 0]."""
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4

# Diferenciální evoluce (DE) - algoritmus
class DE:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=30, max_evaluations=3000, F=0.5, CR=0.9):
        # Inicializace parametrů pro algoritmus DE
        self.func = func  # Cílová funkce, kterou chceme minimalizovat
        self.dimension = dimension  # Dimenze vyhledávacího prostoru
        self.lower_bound = lower_bound  # Dolní hranice vyhledávacího prostoru
        self.upper_bound = upper_bound  # Horní hranice vyhledávacího prostoru
        self.population_size = population_size  # Počet jedinců v populaci
        self.max_evaluations = max_evaluations  # Maximální počet vyhodnocení funkce
        self.F = F  # Diferenciální váha pro mutaci
        self.CR = CR  # Pravděpodobnost křížení

    def optimize(self):
        # Inicializace populace náhodně v rámci hranic
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        fitness = np.array([self.func(ind) for ind in population])  # Vyhodnocení počáteční fitness
        evaluations = len(fitness)  # Počet vyhodnocení funkce

        # Hlavní optimalizační smyčka
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                # Výběr tří odlišných jedinců náhodně z populace (kromě aktuálního jedince)
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Krok mutace: vytvoření mutovaného vektoru
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Krok křížení: vytvoření testovacího vektoru
                trial = np.array([mutant[j] if np.random.rand() < self.CR else population[i, j] for j in range(self.dimension)])

                # Vyhodnocení testovacího vektoru
                trial_fitness = self.func(trial)
                evaluations += 1

                # Krok selekce: nahrazení aktuálního jedince, pokud je testovací vektor lepší
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        # Vrátit nejlepší nalezené řešení
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

# Optimalizace pomocí částicového roje (PSO)
class PSO:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=30, max_evaluations=3000, w=0.5, c1=1.5, c2=1.5):
        # Inicializace parametrů pro algoritmus PSO
        self.func = func  # Cílová funkce, kterou chceme minimalizovat
        self.dimension = dimension  # Dimenze vyhledávacího prostoru
        self.lower_bound = lower_bound  # Dolní hranice vyhledávacího prostoru
        self.upper_bound = upper_bound  # Horní hranice vyhledávacího prostoru
        self.population_size = population_size  # Počet částic v roji
        self.max_evaluations = max_evaluations  # Maximální počet vyhodnocení funkce
        self.w = w  # Inerciální váha
        self.c1 = c1  # Kognitivní koeficient
        self.c2 = c2  # Sociální koeficient

    def optimize(self):
        # Inicializace částic a jejich rychlostí náhodně v rámci hranic
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dimension))
        fitness = np.array([self.func(ind) for ind in population])  # Vyhodnocení počáteční fitness
        personal_best = population.copy()  # Nejlepší známé pozice každé částice
        personal_best_fitness = fitness.copy()  # Fitness hodnoty osobních nejlepších pozic
        global_best = population[np.argmin(fitness)]  # Nejlepší známá pozice v roji
        global_best_fitness = np.min(fitness)  # Fitness hodnota globálně nejlepší pozice
        evaluations = len(fitness)  # Počet vyhodnocení funkce

        # Hlavní optimalizační smyčka
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                # Aktualizace rychlosti částice
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * np.random.rand() * (personal_best[i] - population[i]) +
                    self.c2 * np.random.rand() * (global_best - population[i])
                )

                # Aktualizace pozice částice a její omezení v rámci hranic
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Vyhodnocení nové pozice částice
                fitness[i] = self.func(population[i])
                evaluations += 1

                # Aktualizace osobních a globálních nejlepších pozic, pokud je nová pozice lepší
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness[i]

                    if fitness[i] < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness[i]

        # Vrátit nejlepší nalezené řešení
        return global_best, global_best_fitness

# Samoorganizující se migrační algoritmus (SOMA)
class SOMA:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=30, max_evaluations=3000, path_length=3.0, step=0.11):
        # Inicializace parametrů pro algoritmus SOMA
        self.func = func  # Cílová funkce, kterou chceme minimalizovat
        self.dimension = dimension  # Dimenze vyhledávacího prostoru
        self.lower_bound = lower_bound  # Dolní hranice vyhledávacího prostoru
        self.upper_bound = upper_bound  # Horní hranice vyhledávacího prostoru
        self.population_size = population_size  # Počet jedinců v populaci
        self.max_evaluations = max_evaluations  # Maximální počet vyhodnocení funkce
        self.path_length = path_length  # Parametr délky cesty
        self.step = step  # Krokový parametr

    def optimize(self):
        # Inicializace populace náhodně v rámci hranic
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        fitness = np.array([self.func(ind) for ind in population])  # Vyhodnocení počáteční fitness
        leader_idx = np.argmin(fitness)  # Index nejlepšího jedince (vůdce)
        leader = population[leader_idx]  # Nejlepší jedinec
        evaluations = len(fitness)  # Počet vyhodnocení funkce

        # Hlavní optimalizační smyčka
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                if i == leader_idx:  # Přeskočení vůdce
                    continue

                # Pohyb směrem k vůdci podél cesty
                for t in np.arange(0, self.path_length, self.step):
                    candidate = population[i] + t * (leader - population[i])
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)  # Omezení v rámci hranic
                    candidate_fitness = self.func(candidate)  # Vyhodnocení kandidáta
                    evaluations += 1

                    # Aktualizace pozice jedince, pokud je kandidát lepší
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness

            # Aktualizace vůdce
            leader_idx = np.argmin(fitness)
            leader = population[leader_idx]

        # Vrátit nejlepší nalezené řešení
        return leader, fitness[leader_idx]

# Algoritmus světlušek (FA)
class FA:
    def __init__(self, func, dimension, lower_bound, upper_bound, population_size=30, max_evaluations=3000, alpha=0.5, beta=1.0, gamma=1.0):
        # Inicializace parametrů pro algoritmus FA
        self.func = func  # Cílová funkce, kterou chceme minimalizovat
        self.dimension = dimension  # Dimenze vyhledávacího prostoru
        self.lower_bound = lower_bound  # Dolní hranice vyhledávacího prostoru
        self.upper_bound = upper_bound  # Horní hranice vyhledávacího prostoru
        self.population_size = population_size  # Počet světlušek v populaci
        self.max_evaluations = max_evaluations  # Maximální počet vyhodnocení funkce
        self.alpha = alpha  # Parametr náhodnosti
        self.beta = beta  # Atraktivita na vzdálenost 0
        self.gamma = gamma  # Koeficient absorpce světla

    def optimize(self):
        # Inicializace populace náhodně v rámci hranic
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))
        fitness = np.array([self.func(ind) for ind in population])  # Vyhodnocení počáteční fitness
        evaluations = len(fitness)  # Počet vyhodnocení funkce

        # Hlavní optimalizační smyčka
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # Pohyb směrem k jasnější světlušce
                        r = np.linalg.norm(population[i] - population[j])  # Vzdálenost mezi světluškami
                        beta_effective = self.beta * np.exp(-self.gamma * r**2)  # Upravená atraktivita
                        # Aktualizace pozice světlušky
                        population[i] += beta_effective * (population[j] - population[i]) + self.alpha * np.random.uniform(-1, 1, self.dimension)
                        population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)  # Omezení v rámci hranic
                        fitness[i] = self.func(population[i])  # Vyhodnocení nové pozice
                        evaluations += 1

        # Vrátit nejlepší nalezené řešení
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

# Spuštění experimentů a uložení výsledků
if __name__ == "__main__":
    # Definice testovacích funkcí a optimalizačních algoritmů
    test_functions = {
        "Sphere": sphere,
        "Ackley": ackley,
        "Rastrigin": rastrigin,
        "Rosenbrock": rosenbrock,
        "Griewank": griewank,
        "Schwefel": schwefel,
        "Levy": levy,
        "Michalewicz": michalewicz,
        "Zakharov": zakharov
    }

    algorithms = {
        "DE": DE,
        "PSO": PSO,
        "SOMA": SOMA,
        "FA": FA
    }

    # Parametry experimentu
    dimension = 30
    lower_bound = -100
    upper_bound = 100
    population_size = 30
    max_evaluations = 3000
    num_experiments = 30

    # Spuštění experimentů pro každou testovací funkci a algoritmus
    for func_name, func in test_functions.items():
        experiment_results = []

        for algorithm_name, algorithm in algorithms.items():
            algorithm_results = []
            for experiment in range(1, num_experiments + 1):
                optimizer = algorithm(func, dimension, lower_bound, upper_bound, population_size, max_evaluations)
                _, best_fitness = optimizer.optimize()
                algorithm_results.append(best_fitness)

            mean_value = np.mean(algorithm_results)  # Výpočet průměrné fitness
            std_dev = np.std(algorithm_results)  # Výpočet směrodatné odchylky fitness

            experiment_results.append([algorithm_name] + algorithm_results + [mean_value, std_dev])

        # Uložení výsledků do Excel souboru
        columns = ["Algorithm"] + [f"Experiment {i}" for i in range(1, num_experiments + 1)] + ["Mean", "Std Dev"]
        df = pd.DataFrame(experiment_results, columns=columns)
        df.to_excel(f"{func_name}_results.xlsx", index=False)

    print("Všechny výsledky byly úspěšně uloženy do Excel souborů.")
