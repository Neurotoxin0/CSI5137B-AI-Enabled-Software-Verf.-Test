"""
@Description  :   GA Solver
@Author1      :   Yang Xu, 300342009
@Author2      :   XXX, XXX
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
        pass


    def assess_fitness(self):
        pass

    def initialize_population(self):
        pass

    def evaluate_population(self):
        pass

    def select_parents(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def get_best_individual(self):
        pass