import copy, random
from tqdm import tqdm

from models.general import *
from models.prototype import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: frozenset['Truck'], iterations: int = 1000) -> None:
        """
        Initialize Random Search algorithm.

        Parameters:
        - problem_instance (DeliveryProblem): The initial problem instance.
        - truck_types (frozenset): Set of available truck types.
        - iterations (int): Number of iterations for the search.
        """
        super().__init__(problem_instance, truck_types=list(truck_types))  # Convert frozenset to list
        self.iterations = iterations

    def search(self) -> 'DeliveryProblem':
        """
        Perform Random Search to optimize the solution.

        Returns:
        - DeliveryProblem: The best solution found during the search.
        """
        current_solution = self.__generate_random_solution()
        best_solution = current_solution
        best_cost = self._evaluate_solution(current_solution)

        if self.debug: print("Running Random Search...")
        progress_bar = tqdm(range(self.iterations), desc="Random Search Progress", position = 0, leave=False)

        for i in progress_bar:
            # Generate a new neighbor solution
            neighbor_solution = self.__generate_random_solution()
            neighbor_cost = self._evaluate_solution(neighbor_solution)

            # Check if the new solution is better
            if neighbor_cost < best_cost:
                best_solution = neighbor_solution
                best_cost = neighbor_cost

            # Update progress bar description
            progress_bar.set_postfix({"Best Cost": best_cost})

        return best_solution

    def __generate_random_solution(self) -> 'DeliveryProblem':
        """
        Generate a random solution with valid truck assignments and randomized routes.

        Returns:
        - DeliveryProblem: A feasible solution.
        """
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []  # Clear previous routes

        # Assign orders to trucks and create valid routes
        for order in solution.orders:
            valid_trucks = [truck for truck in self.truck_types if truck.can_load(order)]
            if valid_trucks:
                selected_truck = random.choice(valid_trucks).copy()
                solution._DeliveryProblem__assign_order_to_truck(order, truck=selected_truck)

        # Shuffle orders within routes while ensuring constraints
        for route in solution.routes:
            shuffled_orders = self.__shuffle_valid_orders(route)
            route.orders = shuffled_orders
            route.calculate_route_details()

        return solution

    def __shuffle_valid_orders(self, route: 'Route') -> list:
        """
        Shuffle orders in a route while ensuring no "teleportation" occurs.

        Parameters:
        - route (Route): The current delivery route.

        Returns:
        - list: A valid list of shuffled orders.
        """
        orders = route.orders[:]
        random.shuffle(orders)  # Shuffle orders randomly

        # Validate the shuffled route
        valid_orders = []
        current_location = orders[0].start_city

        for order in orders:
            if current_location == order.start_city:  # Ensure continuity of the route
                valid_orders.append(order)
                current_location = order.end_city  # Move to next location
            else:
                break  # Invalid shuffle, stop and return the original orders

        return valid_orders if len(valid_orders) == len(orders) else route.orders
