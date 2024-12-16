import os, sys
from abc import ABC, abstractmethod

Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
Path = Path.replace("models/", "")
sys.path.append(Path)

import config
from models.general import *

# SearchAlgorithm class (abstract base class)
class SearchAlgorithm(ABC):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck']) -> None:
        """
        Initialize the search algorithm with the given problem instance.

        Parameters:
        problem_instance (DeliveryProblem): The problem instance to be solved by the algorithm.
        truck_types (list): List of truck types available for the delivery problem.
        
        Returns:
        None
        """
        self.problem_instance = problem_instance.copy()
        self.truck_types = truck_types
        
        #self.debug = config.debug
        self.debug = False


    @abstractmethod
    def search(self) -> 'DeliveryProblem':
        """
        Execute the algorithm to find the best solution.

        Parameters:
        None

        Returns:
        DeliveryProblem: The best solution found by the algorithm.
        """
        #return self.problem_instance
        pass


    def _evaluate_solution(self, solution: 'DeliveryProblem') -> float:
        """
        Evaluate the quality of a solution.

        Parameters:
        solution (DeliveryProblem): The solution to evaluate.

        Returns:
        float: The evaluation score (lower is better, typically cost, distance, or fuel).
        """
        total_cost = 0
        for route in solution.routes:
            total_cost += route.total_cost  # Assuming each route has a `total_cost` property
        return total_cost
