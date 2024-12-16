import random
import copy
from tqdm import tqdm
from models.general import *
from models.prototype import SearchAlgorithm


class GeneticAlgorithm(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'],
                 population_size: int = 50, mutation_rate: float = 0.2, crossover_rate: float = 0.8) -> None:
        super().__init__(problem_instance, truck_types=truck_types)
        self.population_size = population_size
        self.generations = config.iterations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def search(self) -> 'DeliveryProblem':
        """
        Perform the Genetic Algorithm to find the best solution.
        """
        population = self.__initialize_population()
        best_solution = min(population, key=self._evaluate_solution)
        best_cost = self._evaluate_solution(best_solution)

        if self.debug: print("Running Genetic Algorithm...")
        for generation in tqdm(range(self.generations), desc="Genetic Algorithm Progress", position=1, leave=False):
            fitness_scores = [(individual, self._evaluate_solution(individual)) for individual in population]
            parents = self.__select_parents(fitness_scores)
            offspring = self.__generate_offspring(parents)
            population = self.__select_next_generation(fitness_scores, offspring)

            current_best = min(population, key=self._evaluate_solution)
            current_best_cost = self._evaluate_solution(current_best)

            if current_best_cost < best_cost:
                best_solution = current_best
                best_cost = current_best_cost

            if generation % 10 == 0 and self.debug:
                print(f"Generation {generation + 1}: Best Cost = {best_cost}")

        return best_solution

    def __initialize_population(self) -> list:
        return [self.__generate_random_solution() for _ in range(self.population_size)]

    def __generate_random_solution(self) -> 'DeliveryProblem':
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []

        for order in solution.orders:
            valid_trucks = [truck for truck in self.truck_types if truck.can_load(order)]
            if valid_trucks:
                selected_truck = random.choice(valid_trucks).copy()
                solution._DeliveryProblem__assign_order_to_truck(order, truck=selected_truck)

        for route in solution.routes:
            random.shuffle(route.orders)
            route.calculate_route_details()

        return solution

    def __select_parents(self, fitness_scores: list) -> list:
        return [min(random.sample(fitness_scores, 5), key=lambda x: x[1])[0] for _ in range(self.population_size)]

    def __generate_offspring(self, parents: list) -> list:
        offspring = []
        for _ in range(len(parents) // 2):
            if random.random() < self.crossover_rate:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.__crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend(random.sample(parents, 2))
        return [self.__mutate(ind) for ind in offspring]

    def __crossover(self, parent1: 'DeliveryProblem', parent2: 'DeliveryProblem') -> tuple:
        """
        Perform multi-point crossover between two parents.
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Multi-point crossover
        for i in range(0, len(parent1.orders), 2):  # Swap orders at even indices
            self.__swap_orders(child1, child2, parent1.orders[i], parent2.orders[i])

        return child1, child2


    def __mutate(self, individual: 'DeliveryProblem') -> 'DeliveryProblem':
        """
        Apply mutation by swapping random orders in a route.
        """
        mutated = copy.deepcopy(individual)

        for route in mutated.routes:
            if random.random() < self.mutation_rate:
                if len(route.orders) > 1:
                    # Swap two random orders in the route
                    i, j = random.sample(range(len(route.orders)), 2)
                    route.orders[i], route.orders[j] = route.orders[j], route.orders[i]
                    route.calculate_route_details()
        return mutated


    def __swap_orders(self, child1, child2, order1, order2):
        for route in child1.routes:
            if order1 in route.orders:
                route.orders.remove(order1)
                route.orders.append(order2)
                break
        for route in child2.routes:
            if order2 in route.orders:
                route.orders.remove(order2)
                route.orders.append(order1)
                break

    def __select_next_generation(self, fitness_scores: list, offspring: list) -> list:
        """
        Select the next generation from the current population and offspring.
        """
        # Combine the current population and offspring
        combined_population = [individual for individual, _ in fitness_scores] + offspring
        
        # Sort by fitness
        combined_population.sort(key=self._evaluate_solution)
        
        # Retain the best individual (elite) and fill the rest
        next_generation = [combined_population[0]]  # Elite individual
        next_generation += random.sample(combined_population[1:], self.population_size - 1)
        return next_generation

