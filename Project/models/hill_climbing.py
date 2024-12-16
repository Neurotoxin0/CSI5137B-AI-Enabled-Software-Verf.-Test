import copy
import random
from tqdm import tqdm

from models.general import *
from models.prototype import SearchAlgorithm


class HillClimbing(SearchAlgorithm):
    def search(self) -> 'DeliveryProblem':
        """
        Perform the hill climbing search to find the best solution.

        Hill climbing first focuses on optimizing truck assignments and then optimizes
        route assignments. It iteratively explores the solution space by generating
        neighbors and selecting better solutions.

        Returns:
        - DeliveryProblem: The best solution found after the optimization process.
        """
        # Initial solution
        current_solution = self.problem_instance
        best_solution = current_solution

        # Step 1: Optimize truck assignments
        if self.debug:
            print("Optimizing truck assignments...")
        for _ in tqdm(range(config.iterations), desc='Hill Climbing Process (Truck Assignment)', position=1, leave=False):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=True)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        # Step 2: Optimize route assignments
        if self.debug:
            print("Optimizing route assignments...")
        for _ in tqdm(range(config.iterations), desc='Hill Climbing Process (Route Optimization)', position=1, leave=False):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=False)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        return best_solution

    def __generate_neighbor(self, current_solution: 'DeliveryProblem', optimize_truck: bool) -> 'DeliveryProblem':
        """
        Generate a neighboring solution by applying one of the following strategies:
        1. Modify truck assignments.
        2. Optimize order delivery sequence within a route.
        3. Randomly reassign an order to a different truck.

        Parameters:
        - current_solution (DeliveryProblem): The current solution to modify.
        - optimize_truck (bool): Flag to prioritize truck optimization.

        Returns:
        - DeliveryProblem: A new neighboring solution.
        """
        # Create a deep copy to avoid modifying the current solution
        neighbor_solution = copy.deepcopy(current_solution)

        # Introduce a mix of strategies based on random probability
        random_choice = random.random()

        if optimize_truck:
            # If truck optimization is prioritized
            if random_choice < 0.6:
                self.__modify_truck_assignments(neighbor_solution)  # Modify truck assignments
            else:
                self.__reassign_order_to_different_truck(neighbor_solution)  # Reassign orders to other trucks
        else:
            # If route optimization is prioritized
            if random_choice < 0.6:
                self.__optimize_order_assignments(neighbor_solution)  # Optimize order sequence
            else:
                self.__reassign_order_to_different_truck(neighbor_solution)

        return neighbor_solution

    def __optimize_order_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Optimize the delivery order within a randomly selected route by swapping multiple pairs of orders.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        if not solution.routes:
            return

        # Randomly choose a route
        chosen_route = random.choice(solution.routes)
        route_orders = chosen_route.orders

        if len(route_orders) > 1:
            # Perform up to 3 random swaps to explore a larger neighborhood
            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                i, j = random.sample(range(len(route_orders)), 2)
                route_orders[i], route_orders[j] = route_orders[j], route_orders[i]

            # Update the route after reordering
            chosen_route.orders = route_orders
            chosen_route.calculate_route_details()

    def __modify_truck_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Modify the truck assignment for a random route by selecting a compatible truck.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        if not solution.routes:
            return

        # Select a random route
        chosen_route = random.choice(solution.routes)
        chosen_truck = chosen_route.truck
        cargos = chosen_truck.cargo

        if len(cargos) == 0:
            return  # Skip if there are no orders

        # Find compatible trucks that can carry the same cargo
        compatible_trucks = [
            t for t in self.truck_types
            if t.truck_id != chosen_truck.truck_id and t.truck_capacity >= sum(cargo.weight for cargo in cargos)
            and t.truck_size >= sum(cargo.area for cargo in cargos)
        ]
        if compatible_trucks:
            new_truck = random.choice(compatible_trucks).copy()
            for cargo in cargos:
                new_truck.load_cargo(cargo)

            # Replace the truck in the chosen route
            chosen_route.truck = new_truck
            chosen_route.calculate_route_details()

    def __reassign_order_to_different_truck(self, solution: 'DeliveryProblem') -> None:
        """
        Reassign a randomly selected order to a different truck.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        if not solution.routes:
            return

        # Select a random route and order
        chosen_route = random.choice(solution.routes)
        if not chosen_route.orders:
            return

        order_to_move = random.choice(chosen_route.orders)
        chosen_route.orders.remove(order_to_move)
        chosen_route.truck.unload_cargo(order_to_move)
        chosen_route.calculate_route_details()

        # Reassign to a new valid truck
        valid_trucks = [t for t in self.truck_types if t.can_load(order_to_move)]
        if valid_trucks:
            new_truck = random.choice(valid_trucks).copy()
            solution._DeliveryProblem__assign_order_to_truck(order_to_move, truck=new_truck)
