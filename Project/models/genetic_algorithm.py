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
        """
        super().__init__(problem_instance, truck_types=truck_types)
        self.population_size = population_size
        self.generations = config.iterations  # Number of generations (iterations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def search(self) -> 'DeliveryProblem':
        """
        Perform the Genetic Algorithm to optimize the solution.
        """
        # Step 1: Initialize the population
        population = self.__initialize_population()
        best_solution = min(population, key=self._evaluate_solution)  # Best individual in the population
        best_cost = self._evaluate_solution(best_solution)

        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            progress_bar = tqdm(total=self.generations, desc="Genetic Algorithm Progress")

            for generation in range(self.generations):
                # Submit generation tasks to thread pool
                future = executor.submit(self._run_generation, population)
                result = future.result()
                population, generation_best_cost, generation_best_solution = result

                # Update best solution
                if generation_best_cost < best_cost:
                    best_solution = generation_best_solution
                    best_cost = generation_best_cost

                progress_bar.update(1)
                if generation % 10 == 0:  # Optional: Log progress every 10 generations
                    print(f"Generation {generation}: Best Cost = {best_cost}")

            progress_bar.close()

        return best_solution

    def __initialize_population(self) -> list:
        """
        Initialize a population of solutions.
        """
        return [self.__generate_random_solution() for _ in range(self.population_size)]

    def __generate_random_solution(self) -> 'DeliveryProblem':
        """
        Generate a random solution.
        """
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []

        for order in solution.orders:
            # Assign orders to trucks
            valid_trucks = [truck for truck in self.truck_types if truck.can_load(order)]
            if valid_trucks:
                truck = random.choice(valid_trucks).copy()
                solution._DeliveryProblem__assign_order_to_truck(order, truck=truck)

        for route in solution.routes:
            random.shuffle(route.orders)
            route.calculate_route_details()

        return solution

    @staticmethod
    def _run_generation(population: list) -> tuple:
        """
        Perform a single generation of the genetic algorithm.
        
        Parameters:
        - population (list): Current population of solutions.
        
        Returns:
        - tuple: (new_population, best_cost, best_solution)
        """
        # Evaluate fitness of the current population
        fitness_scores = [(ind, GeneticAlgorithm._evaluate_solution_static(ind)) for ind in population]

        # Select parents
        parents = GeneticAlgorithm.__select_parents(fitness_scores)

        # Generate offspring
        offspring = GeneticAlgorithm.__generate_offspring(parents)

        # Combine parents and offspring for next generation
        new_population = GeneticAlgorithm.__select_next_generation(fitness_scores, offspring)

        # Determine the best solution
        best_solution = min(new_population, key=GeneticAlgorithm._evaluate_solution_static)
        best_cost = GeneticAlgorithm._evaluate_solution_static(best_solution)

        return new_population, best_cost, best_solution

    @staticmethod
    def _evaluate_solution_static(solution: 'DeliveryProblem') -> float:
        """
        Static method to evaluate a solution. Used for multiprocessing compatibility.
        """
        total_cost = 0
        for route in solution.routes:
            total_cost += route.total_cost
        return total_cost

    @staticmethod
    def __select_parents(fitness_scores: list) -> list:
        """
        Perform tournament selection.
        """
        return [min(random.sample(fitness_scores, 5), key=lambda x: x[1])[0] for _ in range(len(fitness_scores))]

    @staticmethod
    def __generate_offspring(parents: list) -> list:
        """
        Generate offspring through crossover and mutation.
        """
        offspring = []
        for _ in range(len(parents) // 2):
            if random.random() < 0.8:  # Crossover probability
                p1, p2 = random.sample(parents, 2)
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                GeneticAlgorithm.__mutate(c1)
                GeneticAlgorithm.__mutate(c2)
                offspring.extend([c1, c2])
        return offspring

    @staticmethod
    def __mutate(solution: 'DeliveryProblem') -> None:
        """
        Apply mutation to a solution.
        """
        for route in solution.routes:
            if random.random() < 0.2:  # Mutation probability
                if len(route.orders) > 1:
                    orders = route.orders[:]
                    random.shuffle(orders)
                    route.orders = orders
                    route.calculate_route_details()

    @staticmethod
    def __select_next_generation(fitness_scores: list, offspring: list) -> list:
        """
        Select the next generation based on fitness.
        """
        combined = [individual for individual, _ in fitness_scores] + offspring
        return sorted(combined, key=GeneticAlgorithm._evaluate_solution_static)[:len(fitness_scores)]
