import copy
import random
from tqdm import tqdm

from models.general import *
from models.prototype import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'], iterations: int = 1000) -> None:
        """
        Initialize the Random Search algorithm.

        Parameters:
        - problem_instance (DeliveryProblem): The problem instance to solve.
        - truck_types (list): List of available truck types.
        - iterations (int): Number of iterations for the search.
        """
        super().__init__(problem_instance, truck_types=truck_types)
        self.iterations = iterations

    def search(self) -> 'DeliveryProblem':
        """
        Perform Random Search to find the best solution.
        """
        # Generate the initial valid solution
        current_solution = self.__generate_valid_solution()
        best_solution = current_solution
        best_cost = self._evaluate_solution(current_solution)

        print("Running Random Search...")
        for i in tqdm(range(self.iterations), desc="Random Search Progress", position=0, leave=True):
            # Generate a valid neighboring solution
            neighbor = self.__generate_valid_solution()
            neighbor_cost = self._evaluate_solution(neighbor)

            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

            if i % 10 == 0:
                print(f"Iteration {i + 1}: Best Cost = {best_cost}")

        return best_solution

    def __generate_valid_solution(self) -> 'DeliveryProblem':
        """
        Generate a valid solution ensuring:
        - Trucks only carry cargo from the same order ID.
        - Routes have fixed start and end points but allow intermediate paths to shuffle.
        - All deliveries are completed before the order's deadline.
        """
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []

        # Step 1: Group orders by order_id
        order_groups = {}
        for order in solution.orders:
            if order.order_id not in order_groups:
                order_groups[order.order_id] = []
            order_groups[order.order_id].append(order)

        # Step 2: Assign orders to trucks and construct routes
        for order_id, orders in order_groups.items():
            remaining_orders = orders.copy()

            while remaining_orders:
                selected_truck = random.choice(self.truck_types).copy()
                current_orders = []

                # Load as much cargo as possible from the same order_id
                for order in remaining_orders[:]:
                    if selected_truck.can_load(order):
                        selected_truck.load_cargo(order)
                        current_orders.append(order)
                        remaining_orders.remove(order)

                # If orders were loaded, create a valid route
                if current_orders:
                    route = self.__generate_valid_route(selected_truck, current_orders, solution.city_manager)
                    if route:
                        solution.routes.append(route)

        return solution

    def __generate_valid_route(self, truck: 'Truck', orders: list['Order'], city_manager: 'CityManager') -> 'Route':
        """
        Generate a valid route for a truck with the given orders.
        The route starts and ends at the fixed order points but allows intermediate shuffling.

        Parameters:
        - truck (Truck): The truck carrying the orders.
        - orders (list): List of orders to deliver.
        - city_manager (CityManager): City manager for distance calculations.

        Returns:
        - Route: A valid route if feasible, else None.
        """
        # Fixed start and end cities
        start_city = orders[0].start_city
        end_city = orders[0].end_city

        # Generate intermediate cities to visit
        intermediate_cities = [order.end_city for order in orders if order.end_city != start_city and order.end_city != end_city]
        random.shuffle(intermediate_cities)

        # Create a valid route
        full_route = [start_city] + intermediate_cities + [end_city]
        total_distance = 0
        total_time = 0
        truck_copy = copy.deepcopy(truck)

        # Validate the route (ensure no "instant jumps" and meets deadlines)
        for i in range(len(full_route) - 1):
            distance = city_manager.distance_between_cities(city1=full_route[i], city2=full_route[i + 1])
            total_distance += distance
            travel_time = distance / truck_copy.truck_speed
            total_time += travel_time

            # Check delivery deadlines
            if total_time > (orders[0].end_time - orders[0].start_time).total_seconds() / 3600:
                return None  # Invalid route if deadline is missed

        # Create the new route
        truck_copy.remaining_capacity = truck.truck_capacity - sum(order.weight for order in orders)
        truck_copy.remaining_area = truck.truck_size - sum(order.area for order in orders)

        route = Route(truck_copy, orders, city_manager)
        return route
