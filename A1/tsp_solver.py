"""
@Description  :   GA Solver
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import math, random, string
from concurrent.futures import ProcessPoolExecutor, as_completed


debug = False


def euclidean_distance(city1: tuple, city2: tuple) -> float:
    """
    Calculate the Euclidean distance between two cities.

    Parameters:
    - city1 (tuple): A tuple containing the x and y coordinates of the first city.
    - city2 (tuple): A tuple containing the x and y coordinates of the second city.

    Returns:
    - distance (float): The Euclidean distance between the two cities.
    """
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def verify_tsp_solution(solution: list, num_nodes: int) -> tuple:
    """
    Verify if the solution is valid for the Travelling Salesman Problem (TSP).

    Parameters:
    - solution (list): A list of city indices representing the tour.
    - num_nodes (int): The number of cities in the TSP instance.

    Returns:
    - valid (bool): True if the solution is valid, False otherwise.
    - message (str): A message indicating the validation result.
    """
    # Check if the solution has the correct number of cities
    if len(solution) != num_nodes:
        return False, f"The solution contains {len(solution)} nodes, but it should contain {num_nodes} nodes."

    # Check if all cities are within the valid range
    if not all(1 <= city <= num_nodes for city in solution):
        invalid_cities = [city for city in solution if city < 1 or city > num_nodes]
        return False, f"The solution contains invalid city indices: {invalid_cities}. Valid range is 1 to {num_nodes}."

    # Check for duplicates and missing cities
    city_set = set(solution)
    if len(city_set) != num_nodes:
        missing_cities = [city for city in range(1, num_nodes + 1) if city not in city_set]
        duplicate_cities = [city for city in solution if solution.count(city) > 1]
        
        if missing_cities and duplicate_cities:
            return False, f"The solution is missing cities: {missing_cities} and contains duplicate cities: {duplicate_cities}."
        elif missing_cities:
            return False, f"The solution is missing cities: {missing_cities}."
        elif duplicate_cities:
            return False, f"The solution contains duplicate cities: {duplicate_cities}."

    # If all checks pass, the solution is valid
    return True, "The solution is valid."


class GeneticAlgorithm:
    """
    A class to represent a genetic algorithm solver for the Travelling Salesman Problem (TSP).

    Attributes:
    - popsize (int): The size of the population.
    - mutation_rate (float): The probability of mutation occurring in offspring.
    - generations (int): The number of generations to run the algorithm.
    - tournament_size (int): The number of individuals in each tournament selection.
    - population (list): The current population of tours.
    - best_individual (list): The best tour found so far.
    - best_fitness (float): The total distance of the best tour found.
    - cities (list): The list of cities with their coordinates and indices.
    
    Methods:
    - _clr(): Clear the genetic algorithm attributes.
    - solve(node_coords): Run the genetic algorithm on the provided cities.
    - assess_fitness(tour): Calculate the total distance for a given tour.
    - initialize_population(): Initialize the population with random tours.
    - evaluate_population(): Evaluate the fitness of the population and update the best solution.
    - select_parents(): Select two parents for crossover using a tournament selection method.
    - crossover(parent1, parent2): Perform ordered crossover to create an offspring.
    - mutate(tour): Perform swap mutation on a tour.
    - get_best_individual(): Return the best tour and its total distance.
    """

    def __init__(self, *, popsize: int, mutation_rate: float, generations: int, tournament_size: int) -> None:
        """
        Initialize the genetic algorithm with the given parameters.

        Parameters:
        - popsize (int): The size of the population.
        - mutation_rate (float): The probability of a mutation occurring during crossover.
        - generations (int): The number of generations to run the genetic algorithm.
        - tournament_size (int): The number of individuals to include in each tournament selection.
        """
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tournament_size = tournament_size  # Store the tournament size

        self._clr()  # Init the genetic algorithm attributes
        

    def _clr(self):
        """
        Clear the genetic algorithm attributes.
        """
        self.population = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.cities = []


    def solve(self, node_coords: list) -> tuple:
        """ 
        Run the genetic algorithm, evolving the population over generations, 
        and return the best solution.

        Parameters:
        - node_coords (list): A list of tuples containing the city indices and coordinates; e.g., [(0, (x0, y0)), (1, (x1, y1)), ...].

        Returns:
        - tuple: The best solution (list of city indices), the total distance of the best solution, and a tuple indicating the validation result.
        """
        self._clr()  # Clear the genetic algorithm attributes

        self.cities = node_coords
        self.initialize_population()  # Initial population of random tours
        self.evaluate_population()  # Assess the fitness of the initial population
        
        for generation in range(self.generations):
            new_population = []
            for _ in range(self.popsize):
                parent1, parent2 = self.select_parents()  # Select two parents using tournament selection
                child = self.crossover(parent1, parent2)  # Perform crossover to produce a child
                self.mutate(child)
                new_population.append(child)

            self.population = new_population
            self.evaluate_population()
            
            # Print progress every 10 generations
            if debug and (generation + 1) % 10 == 0: print(f'Generation {generation + 1}/{self.generations}, Best Distance: {self.best_fitness:.2f}')
             
        solution, fitness = self.get_best_individual()
        return solution, fitness, verify_tsp_solution(solution, len(self.cities))


    def assess_fitness(self, tour: list) -> float:
        """
        Calculate the total distance for a given tour.

        Parameters:
        - tour (list): A list of city indices representing the tour.

        Returns:
        - fitness (float): The total distance of the tour.
        """
        total_distance = 0
        for i in range(len(tour)):
            city1 = self.cities[tour[i]]
            city2 = self.cities[tour[(i + 1) % len(tour)]]  # Connect the last city back to the first
            total_distance += euclidean_distance(city1, city2)
        return total_distance
    

    def initialize_population(self) -> None:
        """ 
        Initialize the population with random tours (permutations of city indices). 
        """
        self.population = []
        num_cities = len(self.cities)
        for _ in range(self.popsize):
            # Create a random tour as a permutation of city indices
            tour = random.sample(range(num_cities), num_cities)
            self.population.append(tour)


    def evaluate_population(self) -> None:
        """ 
        Evaluate the fitness of the entire population and update the best solution if found.
        """
        for individual in self.population:
            fitness = self.assess_fitness(individual)  # Calculate the total distance of the tour
            if fitness < self.best_fitness:
                # Update the best solution if the current one is better
                self.best_fitness = fitness
                self.best_individual = individual


    def select_parents(self) -> tuple:
        """ 
        Select two parents using tournament selection.

        Returns:
        - tuple: Two selected parents for crossover.
        """
        # Use self.tournament_size for the tournament selection
        parent1 = min(random.sample(self.population, self.tournament_size), key=self.assess_fitness)
        parent2 = min(random.sample(self.population, self.tournament_size), key=self.assess_fitness)
        return parent1, parent2


    def crossover(self, parent1: list, parent2: list) -> list:
        """ 
        Perform ordered crossover (OX1) between two parents to create an offspring.

        Parameters:
        - parent1 (list): The first parent tour.
        - parent2 (list): The second parent tour.

        Returns:
        - list: The offspring tour.
        """
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))  # Select two crossover points
        child = [None] * size
        child[start:end] = parent1[start:end]  # Copy a segment from the first parent

        pos = end  # Start filling the child from the end of the copied segment
        for city in parent2:
            if city not in child:
                # Fill the remaining positions with cities from the second parent
                if pos >= size: pos = 0
                child[pos] = city
                pos += 1
        return child


    def mutate(self, tour: list) -> None:
        """ 
        Perform swap mutation on a tour.

        Parameters:
        - tour (list): The tour to mutate.
        """
        if random.random() < self.mutation_rate:
            # Swap two random cities in the tour
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]


    def get_best_individual(self) -> tuple:
        """ 
        Return the best individual (tour) and its fitness.

        Returns:
        - tuple: The best tour (list of indices) and its total distance.
        """
        best_tour_indices = [self.cities[i][0] for i in self.best_individual]  # Get original city indices
        return best_tour_indices, self.best_fitness


class GAOptimizer:
    """
    A class to optimize the hyperparameters of the genetic algorithm using random search with double layer parallelization.

    Attributes:
    - n_iter (int): The number of iterations to run random search.
    - max_outer_workers (int): The maximum number of workers for the outer loop.
    - max_inner_workers (int): The maximum number of workers for the inner loop.
    - best_params (dict): The best hyperparameters found during optimization.
    - best_fitness (float): The total fitness of the best parameters.

    Methods:
    - _run_ga(params, tsp_instance): Run the genetic algorithm for a single set of parameters and a single TSP instance.
    - _evaluate_hyperparams(params, tsp_instances): Evaluate a set of hyperparameters on all TSP instances.
    - optimize(tsp_instances): Perform random search to find the best parameters for the genetic algorithm.
    """

    def __init__(self, *, n_iter: int, max_outer_workers: int, max_inner_workers: int) -> None:
        """
        Initialize the optimizer with the number of iterations for random search.

        Parameters:
        - n_iter (int): Number of iterations to run random search.
        - max_outer_workers (int): Maximum number of workers for the outer loop (default: 5).
        - max_inner_workers (int): Maximum number of workers for the inner loop (default: None).
        """
        self.n_iter = n_iter
        self.max_outer_workers = max_outer_workers
        self.max_inner_workers = max_inner_workers
        self.best_params = None
        self.best_fitness = float('inf')


    def _run_ga(self, params: dict, tsp_instance) -> float:
        """
        Helper function to run GA for a single set of parameters and a single TSP instance.

        Parameters:
        - params (dict): Dictionary containing the parameters for the GA.
        - tsp_instance: The TSP instance to run the GA on.

        Returns:
        - float: The total fitness for this set of parameters.
        """
        ga_instance = GeneticAlgorithm(
            popsize=params['popsize'],
            mutation_rate=params['mutation_rate'],
            generations=params['generations'],
            tournament_size=params['tournament_size']
        )
        _, fitness, _ = ga_instance.solve(tsp_instance.node_coords)
        return fitness


    def _evaluate_hyperparams(self, params: dict, tsp_instances: list, outer_id: str) -> tuple:
        """
        Helper function to evaluate a set of hyperparameters on all TSP instances.

        Parameters:
        - params (dict): Dictionary containing the parameters for the GA.
        - tsp_instances (list): A list of TSP instances to run the GA on.
        - outer_id (str): A unique identifier for this outer loop run.

        Returns:
        - tuple: (params, total_fitness) where total_fitness is the sum of fitness values for this parameter set.
        """
        total_fitness = 0
        # Run GA on all TSP instances in parallel
        with ProcessPoolExecutor(max_workers=self.max_inner_workers) as executor:
            futures = {executor.submit(self._run_ga, params, tsp_instance): tsp_instance for tsp_instance in tsp_instances}

            # Collect results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                fitness = future.result()
                total_fitness += fitness
                print(f"\t{outer_id}: {i}/{len(tsp_instances)} instances completed.")

        return params, total_fitness


    def optimize(self, tsp_instances: list) -> dict:
        """
        Perform random search to find the best parameters for the genetic algorithm using multiprocessing.

        Parameters:
        - tsp_instances (list): A list of TSP instances to use for optimization.

        Returns:
        - best_params (dict): Dictionary containing the best parameters found.
        """
        total_ga_runs = self.n_iter * len(tsp_instances)  # Calculate the total number of GA runs
        completed_ga_runs = 0  # Counter for completed GA runs

        with ProcessPoolExecutor(max_workers=self.max_outer_workers) as executor:
            outer_futures = []
            
            for _ in range(self.n_iter):
                # Randomly select unique name and hyperparameters and create a new outer loop run
                outer_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                params = {
                    'popsize': random.choice([50, 100, 200, 300]),
                    'mutation_rate': random.uniform(0.01, 0.2),
                    'generations': random.choice([100, 200, 500]),
                    'tournament_size': random.choice([5, 7, 10])
                }

                # Submit a task to evaluate hyperparameters
                outer_futures.append(
                    executor.submit(self._evaluate_hyperparams, params, tsp_instances, outer_id)
                )

            # Collect results from outer futures (hyperparameter sets) as they complete
            for future in as_completed(outer_futures):
                params, total_fitness = future.result()

                # Update the total completed runs
                completed_ga_runs += len(tsp_instances)
                overall_progress = (completed_ga_runs / total_ga_runs) * 100
                print(f"Overall progress: {completed_ga_runs}/{total_ga_runs} runs completed ({overall_progress:.2f}%).")

                # Update the best parameters if the current total fitness is better
                if total_fitness < self.best_fitness:
                    self.best_fitness = total_fitness
                    self.best_params = params
                    print(f"New best fitness: {self.best_fitness:.2f} with params: {self.best_params}")

        return self.best_params
