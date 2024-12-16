import copy
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from models.general import *
from models.prototype import SearchAlgorithm


class GeneticAlgorithm(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'],
                 population_size: int = 50, mutation_rate: float = 0.2, crossover_rate: float = 0.8) -> None:
        """
        Initialize the Genetic Algorithm with parameters.

        Parameters:
        - problem_instance (DeliveryProblem): The initial problem instance to optimize.
        - truck_types (list): List of available truck types.
        - population_size (int): The number of individuals in the population.
        - mutation_rate (float): The probability of mutation occurring for an individual.
        - crossover_rate (float): The probability of crossover between two parents.
        """
        super().__init__(problem_instance, truck_types=truck_types)
        self.population_size = population_size
        self.generations = config.iterations  # Number of generations (iterations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def search(self) -> 'DeliveryProblem':
        """
        Perform the Genetic Algorithm to optimize the solution.

        Returns:
        - DeliveryProblem: The best solution found after running the algorithm.
        """
        # Step 1: Initialize the population
        population = self.__initialize_population()
        best_solution = min(population, key=self._evaluate_solution)  # Best individual in the population
        best_cost = self._evaluate_solution(best_solution)

        # Thread pool to parallelize generation computation
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            progress_bar = tqdm(total=self.generations, desc="Genetic Algorithm Progress", position=0, leave=True)

            for generation in range(self.generations):
                # Submit generation processing to the thread pool
                future = executor.submit(self.__run_generation, population)
                population, current_best_cost, current_best = future.result()

                # Update best solution if improvement is found
                if current_best_cost < best_cost:
                    best_solution = current_best
                    best_cost = current_best_cost

                progress_bar.update(1)
                if generation % 10 == 0:
                    print(f"Generation {generation + 1}: Best Cost = {best_cost}")

            progress_bar.close()

        return best_solution

    def __run_generation(self, population):
        """
        Process a single generation: selection, crossover, mutation, and next-generation creation.

        Parameters:
        - population (list): The current generation's population.

        Returns:
        - tuple: Next generation population, best cost, and the best solution.
        """
        # Step 1: Evaluate fitness and select parents
        fitness_scores = [(ind, self._evaluate_solution(ind)) for ind in population]
        parents = self.__select_parents(fitness_scores)

        # Step 2: Generate offspring
        offspring = self.__generate_offspring(parents)

        # Step 3: Form next generation
        next_generation = self.__select_next_generation(fitness_scores, offspring)

        # Step 4: Identify the best solution in this generation
        best_solution = min(next_generation, key=self._evaluate_solution)
        best_cost = self._evaluate_solution(best_solution)

        return next_generation, best_cost, best_solution

    def __initialize_population(self) -> list:
        """
        Initialize a population of solutions by randomly assigning trucks and routes.

        Returns:
        - list: A list of randomly generated DeliveryProblem instances.
        """
        return [self.__generate_random_solution() for _ in range(self.population_size)]

    def __generate_random_solution(self) -> 'DeliveryProblem':
        """
        Generate a random solution by assigning orders to trucks and ensuring valid routes.

        Returns:
        - DeliveryProblem: A valid solution with orders assigned to trucks and routes.
        """
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []

        for order in solution.orders:
            # Select a truck that can load the order
            valid_trucks = [truck for truck in self.truck_types if truck.can_load(order)]
            if valid_trucks:
                truck = random.choice(valid_trucks).copy()
                # Assign a valid route for the truck
                route = self.__generate_valid_route(truck, [order], solution.city_manager)
                if route:
                    solution.routes.append(route)

        return solution

    def __generate_valid_route(self, truck, orders, city_manager) -> Route or None:
        """
        Generate a valid route for a truck and a set of orders, ensuring time constraints.

        Parameters:
        - truck (Truck): Truck to assign orders to.
        - orders (list): List of orders to deliver.
        - city_manager (CityManager): Manages city distances.

        Returns:
        - Route or None: A valid Route instance or None if constraints cannot be met.
        """
        route_cities = [orders[0].start_city, orders[0].end_city]
        random.shuffle(route_cities[1:-1])  # Shuffle intermediate cities but keep start and end fixed
        if self.__is_route_valid(route_cities, truck, orders, city_manager):
            return Route(truck, orders, city_manager)
        return None

    def __is_route_valid(self, route, truck, orders, city_manager) -> bool:
        """
        Check if a route satisfies time constraints.

        Parameters:
        - route (list): The list of cities forming the route.
        - truck (Truck): Truck delivering the orders.
        - orders (list): Orders to validate.
        - city_manager (CityManager): Manages city distances.

        Returns:
        - bool: True if the route meets constraints, False otherwise.
        """
        total_time = 0
        for i in range(len(route) - 1):
            distance = city_manager.distance_between_cities(city1=route[i], city2=route[i + 1])
            total_time += distance / truck.truck_speed  # Time = Distance / Speed
            if total_time > (orders[0].end_time - orders[0].start_time).total_seconds() / 3600:
                return False  # Exceeds allowed delivery time
        return True

    def __select_parents(self, fitness_scores: list) -> list:
        """
        Select parents using tournament selection.

        Parameters:
        - fitness_scores (list): List of individuals and their fitness scores.

        Returns:
        - list: Selected parents.
        """
        return [min(random.sample(fitness_scores, 5), key=lambda x: x[1])[0] for _ in range(self.population_size)]

    def __generate_offspring(self, parents: list) -> list:
        """
        Generate offspring through crossover and mutation.

        Parameters:
        - parents (list): The selected parents for reproduction.

        Returns:
        - list: The offspring population.
        """
        offspring = []
        for _ in range(len(parents) // 2):
            if random.random() < self.crossover_rate:
                p1, p2 = random.sample(parents, 2)
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                self.__mutate(c1)
                self.__mutate(c2)
                offspring.extend([c1, c2])
        return offspring

    def __mutate(self, solution: 'DeliveryProblem') -> None:
        """
        Mutate a solution by reordering orders in the routes.

        Parameters:
        - solution (DeliveryProblem): Solution to mutate.
        """
        for route in solution.routes:
            if random.random() < self.mutation_rate:
                random.shuffle(route.orders)
                route.calculate_route_details()

    def __select_next_generation(self, fitness_scores: list, offspring: list) -> list:
        """
        Select the next generation from the current population and offspring.

        Parameters:
        - fitness_scores (list): Current population fitness scores.
        - offspring (list): New offspring.

        Returns:
        - list: The next generation of individuals.
        """
        combined = [individual for individual, _ in fitness_scores] + offspring
        return sorted(combined, key=self._evaluate_solution)[:self.population_size]
