import copy, random
from tqdm import tqdm

from models.general import *
from models.prototype import SearchAlgorithm


class HillClimbing(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck']) -> None:
        """
        Initialize the hill climbing search algorithm with the given problem instance.

        Parameters:
        problem_instance (DeliveryProblem): The problem instance to be solved by the algorithm.
        truck_types (list): List of truck types available for the delivery problem.

        Returns:
        None
        """
        super().__init__(problem_instance)
        self.truck_types = truck_types
    
    def search(self) -> 'DeliveryProblem':
        """
        Perform the hill climbing search to find the best solution, starting with truck assignment optimization.

        Parameters:
        None

        Returns:
        DeliveryProblem: The best solution found after the optimization process.
        """
        # Initial solution
        current_solution = self.problem_instance
        best_solution = current_solution
        
        # Step 1: Optimize truck assignments
        print("Optimizing truck assignments...")
        for _ in tqdm(range(config.iterations), desc='Truck Assignment Optimization'):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=True)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        # Step 2: Optimize route assignments
        print("Optimizing route assignments...")
        for _ in tqdm(range(config.iterations), desc='Route Assignment Optimization'):
            neighbor = self.__generate_neighbor(current_solution, optimize_truck=False)
            if self._evaluate_solution(neighbor) < self._evaluate_solution(best_solution):
                best_solution = neighbor
            current_solution = neighbor

        return best_solution

    def __generate_neighbor(self, current_solution: 'DeliveryProblem', optimize_truck: bool) -> 'DeliveryProblem':
        """
        Generate a neighboring solution by modifying either truck or route assignments based on the flag.

        Parameters:
        current_solution (DeliveryProblem): The current solution to modify.
        optimize_truck (bool): Flag to decide whether to optimize truck assignments or route orders.

        Returns:
        DeliveryProblem: A new solution that is a neighbor of the current one.
        """
        # Create a deep copy of the current solution to generate the neighbor
        neighbor_solution = copy.deepcopy(current_solution)
        
        if optimize_truck:
            # Step 1: Optimize truck assignments for a random route
            self.__modify_truck_assignments(neighbor_solution)
        else:
            # Step 2: Optimize routes order for a random route
            self.__optimize_order_assignments(neighbor_solution)

        return neighbor_solution

    def __optimize_order_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Optimize the order of a random route by swapping a pair of orders.

        Parameters:
        solution (DeliveryProblem): The current solution to modify.

        Returns:
        None
        """
        chosen_route = random.choice(solution.routes)
        route_orders = chosen_route.orders

        if len(route_orders) > 1:
            # Randomly swap a pair of orders in the route
            i, j = random.sample(range(len(route_orders)), 2)
            route_orders[i], route_orders[j] = route_orders[j], route_orders[i]

            # Update the route in the solution
            chosen_route.orders = route_orders
            chosen_route.calculate_route_details()

    def __modify_truck_assignments(self, solution: 'DeliveryProblem') -> None:
        """
        Swap the truck type of a random truck with another truck type that can carry all the orders.

        Parameters:
        solution (DeliveryProblem): The current solution to modify.

        Returns:
        None
        """
        chosen_route = random.choice(solution.routes)
        chosen_truck = chosen_route.truck
        cargos = chosen_truck.cargo

        if len(cargos) <= 1:
            return  # Skip if the truck has only one order

        available_trucks = [t for t in self.truck_types if t.truck_type != chosen_truck.truck_type]
        random.shuffle(available_trucks)

        for candidate_truck in available_trucks:
            total_weight = sum(cargo.weight for cargo in cargos)
            total_area = sum(cargo.area for cargo in cargos)

            if candidate_truck.truck_capacity >= total_weight and candidate_truck.truck_size >= total_area:
                new_truck = candidate_truck.copy()
                for cargo in cargos:
                    new_truck.load_cargo(cargo)

                # Replace the current truck with the new truck in the solution
                chosen_route.truck = new_truck
                chosen_route.calculate_route_details()
                break
