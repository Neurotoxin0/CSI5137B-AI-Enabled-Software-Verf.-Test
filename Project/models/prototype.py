from abc import ABC, abstractmethod

from models.general import *


# SearchAlgorithm class (abstract base class)
class SearchAlgorithm(ABC):
    def __init__(self, problem_instance: 'DeliveryProblem') -> None:
        """
        Initialize the search algorithm with the given problem instance.

        Parameters:
        problem_instance (DeliveryProblem): The problem instance to be solved by the algorithm.
        
        Returns:
        None
        """
        self.problem_instance = problem_instance
        self.best_solution = None  # Will hold the best solution found


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


    @abstractmethod
    def _is_optimal(self, solution: 'DeliveryProblem') -> bool:
        """
        Check if the solution is optimal (based on some stopping condition).

        Parameters:
        solution (DeliveryProblem): The solution to check.

        Returns:
        bool: True if the solution is optimal, False otherwise.
        """
        pass
