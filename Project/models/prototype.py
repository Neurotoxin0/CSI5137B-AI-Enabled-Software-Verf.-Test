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
    def initialize_population(self) -> None:
        """
        Initialize the population for the algorithm (for population-based algorithms).

        Parameters:
        None

        Returns:
        None
        """
        pass


    @abstractmethod
    def select_parents(self) -> None:
        """
        Select the parents from the current population (for genetic algorithms).

        Parameters:
        None

        Returns:
        None
        """
        pass


    @abstractmethod
    def mutate(self) -> None:
        """
        Perform mutation on the selected individuals (for genetic algorithms).

        Parameters:
        None

        Returns:
        None
        """
        pass


    @abstractmethod
    def crossover(self) -> None:
        """
        Perform crossover between selected individuals (for genetic algorithms).

        Parameters:
        None

        Returns:
        None
        """
        pass


    @abstractmethod
    def evaluate_solution(self) -> float:
        """
        Evaluate the quality of a solution.

        Parameters:
        None

        Returns:
        float: The fitness or cost associated with the solution.
        """
        pass
