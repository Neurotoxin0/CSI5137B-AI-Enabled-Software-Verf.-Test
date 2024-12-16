import copy, random
from tqdm import tqdm
from models.general import *
from models.prototype import SearchAlgorithm


class HillClimbing(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck']) -> None:
        """
        Initialize the hill climbing search algorithm.

        Parameters:
        - problem_instance (DeliveryProblem): The problem instance to solve.
        - truck_types (list): List of available truck types.
        """
        super().__init__(problem_instance, truck_types=truck_types)

    def search(self) -> 'DeliveryProblem':
        """
        Perform hill climbing search to optimize truck assignments and route orders.

        Returns:
        - DeliveryProblem: The best solution found.
        """
        current_solution = self.problem_instance
        best_solution = copy.deepcopy(current_solution)
        best_cost = self._evaluate_solution(best_solution)

        # Step 1: Optimize truck assignments
        print("Optimizing truck assignments...")
        for _ in tqdm(range(config.iterations // 2), desc='Hill Climbing (Truck Opt)'):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=True)
            neighbor_cost = self._evaluate_solution(neighbor)

            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

        # Step 2: Optimize route orders
        print("Optimizing route orders...")
        for _ in tqdm(range(config.iterations // 2), desc='Hill Climbing (Route Opt)'):
            neighbor = self.__generate_neighbor(best_solution, optimize_truck=False)
            neighbor_cost = self._evaluate_solution(neighbor)

            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

        return best_solution

    def __generate_neighbor(self, current_solution: 'DeliveryProblem', optimize_truck: bool) -> 'DeliveryProblem':
        """
        Generate a neighboring solution by modifying either truck assignments or route orders.

        Parameters:
        - current_solution (DeliveryProblem): The current solution to modify.
        - optimize_truck (bool): Whether to optimize truck assignments or route orders.

        Returns:
        - DeliveryProblem: A neighboring solution.
        """
        neighbor_solution = copy.deepcopy(current_solution)

        if optimize_truck:
            self.__modify_truck_assignments(neighbor_solution)
        else:
            self.__optimize_route_orders_with_constraints(neighbor_solution)

        return neighbor_solution

    def __modify_truck_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Modify truck assignments for a random route.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        chosen_route = random.choice(solution.routes)
        cargos = chosen_route.truck.cargo

        if len(cargos) == 0: return  # Skip if the truck is empty

        available_trucks = [t for t in self.truck_types if t.truck_capacity >= sum(c.weight for c in cargos) 
                            and t.truck_size >= sum(c.area for c in cargos)]
        if not available_trucks: return

        new_truck = random.choice(available_trucks).copy()
        new_truck.cargo = cargos.copy()
        chosen_route.truck = new_truck
        chosen_route.calculate_route_details()

    def __optimize_route_orders_with_constraints(self, solution: 'DeliveryProblem') -> None:
        """
        Optimize the order of deliveries in a random route while respecting time constraints.

        Parameters:
        - solution (DeliveryProblem): The solution to modify.
        """
        chosen_route = random.choice(solution.routes)
        if len(chosen_route.orders) <= 1: return

        valid_route_found = False
        for _ in range(10):  # Attempt to shuffle orders up to 10 times
            shuffled_orders = chosen_route.orders[:]
            random.shuffle(shuffled_orders)
            
            if self.__is_route_valid(shuffled_orders, chosen_route.truck, solution.city_manager):
                chosen_route.orders = shuffled_orders
                chosen_route.calculate_route_details()
                valid_route_found = True
                break

        if not valid_route_found:
            pass  # If no valid route is found, retain the current order

    def __is_route_valid(self, orders: list['Order'], truck: 'Truck', city_manager: 'CityManager') -> bool:
        """
        Check if the shuffled route satisfies all time constraints.

        Parameters:
        - orders (list): List of orders in the route.
        - truck (Truck): The truck assigned to the route.
        - city_manager (CityManager): Manages city distances.

        Returns:
        - bool: True if the route is valid, False otherwise.
        """
        current_time = orders[0].start_time  # Start at the time of the first order
        current_city = orders[0].start_city

        for order in orders:
            # Travel to the start city of the current order
            travel_time = city_manager.distance_between_cities(city1=current_city, city2=order.start_city) / truck.truck_speed
            current_time += travel_time

            # Check if we arrived too early or too late
            if current_time < order.start_time:
                current_time = order.start_time  # Wait until the start time
            elif current_time > order.end_time:
                return False  # Deadline missed

            # Deliver to the destination city
            delivery_time = city_manager.distance_between_cities(city1=order.start_city, city2=order.end_city) / truck.truck_speed
            current_time += delivery_time

            # Update the current city to the order's end city
            current_city = order.end_city

        return True
