"""
@Description  :   GA Solver
@Author1      :   Yang Xu, 300342009
@Author2      :   XXX, XXX
@Comment      :   Dev with Python 3.10.0
"""


class GeneticAlgorithm:
    def __init__(self, tsp_file):
        self.tsp_file = tsp_file
        self.population = []
        self.best_individual = None
        self.best_cost = None

    def solve(self):
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