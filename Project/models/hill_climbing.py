import copy, random
from tqdm import tqdm

from models.general import *
from models.prototype import SearchAlgorithm


class HillClimbing(SearchAlgorithm):
    """
    Hill Climbing Algorithm to optimize delivery problems. It iteratively improves solutions by 
    modifying truck assignments or optimizing order delivery routes.
    """

    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'], max_workers: int = 4) -> None:
        """
        Initialize the Hill Climbing algorithm with the given problem instance.

        Parameters:
        - problem_instance (DeliveryProblem): The problem instance to solve.
        - truck_types (list): List of available truck types.
        - max_workers (int): Number of workers for parallel execution.
        """
        super().__init__(problem_instance, truck_types=truck_types)
        self.max_workers = max_workers

    def search(self) -> 'DeliveryProblem':
        """
        Perform the Hill Climbing search to optimize truck assignments and delivery routes.

        Returns:
        - DeliveryProblem: The best solution found after optimization.
        """
        current_solution = self.problem_instance
        best_solution = current_solution

        # Step 1: Optimize truck assignments
        if self.debug: print("Optimizing truck assignments...")
        for _ in tqdm(range(config.iterations // 2), desc="Hill Climbing (Truck Opt)", position = 1, leave=False):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=True)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        # Step 2: Optimize route assignments
        if self.debug: print("Optimizing route assignments...")
        for _ in tqdm(range(config.iterations // 2), desc="Hill Climbing (Route Opt)", position = 1, leave=False):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=False)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        return best_solution

    def __generate_neighbor(self, current_solution: 'DeliveryProblem', optimize_truck: bool) -> 'DeliveryProblem':
        """
        Generate a neighboring solution by modifying truck assignments or optimizing routes.

        Parameters:
        - current_solution (DeliveryProblem): The current solution to base modifications on.
        - optimize_truck (bool): Flag to decide whether to optimize trucks or routes.

        Returns:
        - DeliveryProblem: The neighboring solution.
        """
        neighbor_solution = copy.deepcopy(current_solution)

        # Decide to modify trucks or routes
        if optimize_truck:
            self.__modify_truck_assignments(neighbor_solution)
        else:
            self.__optimize_route_orders_with_constraints(neighbor_solution)
        
        return neighbor_solution

    def __modify_truck_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Modify truck assignments for a randomly chosen route to generate a neighboring solution.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        chosen_route = random.choice(solution.routes)
        current_truck = chosen_route.truck

        # Select a truck that can carry all orders in the route
        for candidate_truck in self.truck_types:
            if candidate_truck.truck_type != current_truck.truck_type and all(
                candidate_truck.can_load(order) for order in chosen_route.orders
            ):
                new_truck = candidate_truck.copy()
                for order in chosen_route.orders:
                    new_truck.load_cargo(order)

                # Update the truck in the route
                chosen_route.truck = new_truck
                chosen_route.calculate_route_details()
                break

    def __optimize_route_orders_with_constraints(self, solution: 'DeliveryProblem') -> None:
        """
        Optimize delivery routes by shuffling orders while respecting time constraints.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        chosen_route = random.choice(solution.routes)

        if len(chosen_route.orders) > 1:
            shuffled_orders = chosen_route.orders[:]
            random.shuffle(shuffled_orders)

            # Validate the shuffled route
            if self.__is_route_valid(shuffled_orders, chosen_route.truck, solution.city_manager):
                chosen_route.orders = shuffled_orders
                chosen_route.calculate_route_details()

    def __is_route_valid(self, route_orders: list, truck: 'Truck', city_manager: 'CityManager') -> bool:
        """
        Check if a route is valid based on time constraints and ensures no teleportation.

        Parameters:
        - route_orders (list): List of orders in the route.
        - truck (Truck): The truck assigned to the route.
        - city_manager (CityManager): Manages distances between cities.

        Returns:
        - bool: True if the route is valid, False otherwise.
        """
        if not route_orders:
            return False

        current_city = route_orders[0].start_city  # Starting city of the first order
        total_time = 0

        for order in route_orders:
            # Ensure valid start and end cities
            if current_city != order.start_city:
                return False  # Teleportation detected

            # Calculate travel time
            try:
                travel_time = city_manager.distance_between_cities(city1=current_city, city2=order.end_city) / truck.truck_speed
            except ValueError:
                return False  # Distance unavailable

            total_time += travel_time

            # Check if the delivery violates the time window
            delivery_window = (order.end_time - order.start_time).total_seconds() / 3600
            if total_time > delivery_window:
                return False

            # Update current city to the destination city of the order
            current_city = order.end_city

        return True
