"""
@Description  :   GA Solver
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import math, random


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


class GeneticAlgorithm:
    """
    A class to represent a genetic algorithm solver for the Travelling Salesman Problem (TSP).

    Attributes:
    #TODO

    Methods:
    #TODO
    """

    def __init__(self, popsize: int, mutation_rate: float = 0.1, generations: int = 1000) -> None:
        """
        Initialize the genetic algorithm solver with the given parameters.

        Parameters:
        - popsize (int): The size of the population.
        - mutation_rate (float): The probability of a mutation occurring during crossover.
        - generations (int): The number of generations to run the genetic algorithm.
        
        Returns:
        - None
        """
        self.popsize = popsize
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = []
        self.best_individual = None
        self.cities = []


    def solve(self):
        """ 
        Run the genetic algorithm, evolving the population over generations, 
        and return the best solution.
        """
        # Initialize population and evaluate its fitness
        self.initialize_population()
        self.evaluate_population()
        for generation in range(self.generations):
            new_population = []
            # Generate the next generation
            for _ in range(self.popsize):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
            self.evaluate_population()
            # Print progress every 10 generations
            if (generation + 1) % 10 == 0:
                print(f'Generation {generation+1}/{self.generations}, Best Distance: {self.best_fitness:.2f}')
        return self.get_best_individual()


    def assess_fitness(self):
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
            city2 = self.cities[tour[(i + 1) % len(tour)]]  # wrap-around to the first city
            total_distance += euclidean_distance(city1, city2)
        return total_distance

    def initialize_population(self):
        """ 
        Initialize the population with random tours (permutations of city indices). 
        """
        self.population = []
        for _ in range(self.popsize):
            tour = random.sample(range(len(self.cities)), len(self.cities))
            self.population.append(tour)

    def evaluate_population(self):
        """ 
        Evaluate the fitness of the entire population and update the best solution. 
        """
        for individual in self.population:
            fitness = self.assess_fitness(individual)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual

    def select_parents(self):
        """ 
        Select two parents from the population
        """
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        parent1 = min(tournament, key=self.assess_fitness)
        parent2 = min(random.sample(self.population, tournament_size), key=self.assess_fitness)
        return parent1, parent2

    def crossover(self):
        """ 
        Perform ordered crossover (OX1) between two parents to create an offspring. 
        """
        start, end = sorted(random.sample(range(len(self.cities)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        pos = end
        for city in parent2:
            if city not in child:
                if pos >= len(parent2):
                    pos = 0
                child[pos] = city
                pos += 1
        return child

    def mutate(self):
        """ 
        Perform swap mutation on a tour. 
        """
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]

    def get_best_individual(self):
        """ 
        Return the best individual (tour) and its fitness. 
        """
        return self.best_individual, self.best_fitness