import copy, random

from models.general import *
from models.prototype import SearchAlgorithm


class HillClimbing(SearchAlgorithm):
    def search(self) -> None:
        """
        Perform the hill climbing search to find the best solution.

        Parameters:
        None

        Returns:
        None
        """
        current_solution = self.problem_instance  # Assume this is an initial solution, generated when init the DeliveryProblem
        best_solution = current_solution
        iter_count = 0
        
        while True:
            iter_count += 1
            neighbor = self.__generate_neighbor(current_solution)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor
            if self._is_optimal(current_solution, iter_count):
                break

        return best_solution


    
    def __generate_neighbor(self, current_solution: 'DeliveryProblem') -> 'DeliveryProblem':
        """
        Generate a neighboring solution by slightly modifying the current solution.

        Parameters:
        current_solution (DeliveryProblem): The current solution to modify.

        Returns:
        DeliveryProblem: A new solution that is a neighbor of the current one.
        """
        # Create a deep copy of the current solution to generate the neighbor
        neighbor_solution = copy.deepcopy(current_solution)

        # Step 1: Optimize truck assignments (reassign some orders to different trucks)
        self.__modify_truck_assignments(neighbor_solution)

        # Step 2: Optimize routes for all trucks (minimize the distance and cost)
        self.__optimize_routes(neighbor_solution)

        return neighbor_solution


    def __modify_truck_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Modify the truck assignments by reassigning orders to different trucks.

        Parameters:
        solution (DeliveryProblem): The solution to modify.

        Returns:
        None
        """
        # Randomly swap orders between two trucks
        truck1, truck2 = random.sample(solution.trucks, 2)

        if truck1.cargo and truck2.cargo:
            order1 = random.choice(truck1.cargo)
            order2 = random.choice(truck2.cargo)

            # Swap orders between trucks
            truck1.unload_cargo(order1)
            truck2.unload_cargo(order2)

            truck1.load_cargo(order2)
            truck2.load_cargo(order1)
        

        ''' # Randomly reassign an order to a different truck
        truck = random.choice(solution.trucks)

        if truck.cargo:
            order = random.choice(truck.cargo)
            
            pass'''


    def __optimize_routes(self, solution: 'DeliveryProblem') -> None:
        """
        Optimize the route for each truck in the solution by minimizing the travel distance.

        Parameters:
        solution (DeliveryProblem): The solution with truck assignments to optimize.

        Returns:
        None
        """
        # For each truck, optimize the route (e.g., use a greedy approach or 2-opt)
        for truck in solution.trucks:
            orders = truck.cargo
            optimized_route = self.__find_optimal_route(truck, orders, solution.city_manager)
            truck.route = optimized_route


    def __find_optimal_route(self, truck: 'Truck', orders: list['Order'], city_manager: 'CityManager') -> list['City']:
        """
        Find the optimal route for the truck by optimizing the order of cities to minimize distance.

        Parameters:
        truck (Truck): The truck to optimize the route for.
        orders (list): List of orders assigned to the truck.
        city_manager (CityManager): The CityManager instance to get distances between cities.

        Returns:
        list: The optimized route (list of City objects).
        """
        route = [truck.current_location]  # Start from the truck's current location
        unvisited_orders = orders[:]
        
        while unvisited_orders:
            # Find the closest unvisited order's start city
            nearest_order = min(unvisited_orders, key=lambda order: city_manager.distance_between_cities(city1=route[-1], city2=order.start_city))
            unvisited_orders.remove(nearest_order)
            
            # Add the start city and destination city of the nearest order to the route
            route.append(nearest_order.start_city)
            route.append(nearest_order.end_city)

        return route


    def _is_optimal(self, solution: 'DeliveryProblem', iter_count: int) -> bool:
        """
        Check if the solution is optimal (based on some stopping condition).

        Parameters:
        solution (DeliveryProblem): The solution to check.
        iter_count (int): The current iteration count.

        Returns:
        bool: True if the solution is optimal, False otherwise.
        """
        # TODO: define a more sophisticated stopping condition
        return iter_count >= 100
