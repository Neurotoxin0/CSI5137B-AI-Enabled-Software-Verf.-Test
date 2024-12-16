import random
import copy
from tqdm import tqdm
from models.general import *
from models.prototype import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'], iterations: int = 1000) -> None:
        super().__init__(problem_instance, truck_types=truck_types)
        self.iterations = iterations

    def search(self) -> 'DeliveryProblem':
        """
        Perform Random Search to find the best solution.
        """
        current_solution = self.__generate_random_solution()
        best_solution = current_solution
        best_cost = self._evaluate_solution(current_solution)

        if self.debug: print("Running Random Search...")
        for i in tqdm(range(self.iterations), desc="Random Search Progress", position=4, leave=False):
            neighbor = self.__generate_random_solution()
            neighbor_cost = self._evaluate_solution(neighbor)

            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

            # Log the best cost so far
            if i % 10 == 0 and self.debug:
                print(f"Iteration {i + 1}: Best Cost = {best_cost}")

        return best_solution

    def __generate_random_solution(self) -> 'DeliveryProblem':
        """
        Generate a random solution with truck assignments and shuffled routes.
        """
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
