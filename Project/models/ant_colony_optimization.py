import copy, random
from tqdm import tqdm
from models.general import *
from models.prototype import SearchAlgorithm


class AntColonyOptimization(SearchAlgorithm):
    def __init__(self, problem_instance: 'DeliveryProblem', *, truck_types: list['Truck'],
                 num_ants: int = 50, evaporation_rate: float = 0.1, pheromone_deposit: float = 1.0,
                 alpha: float = 1.0, beta: float = 2.0) -> None:
        """
        Initialize the Ant Colony Optimization algorithm.

        Parameters:
        - problem_instance (DeliveryProblem): The problem instance to solve.
        - truck_types (list): List of available truck types.
        - num_ants (int): Number of ants in the colony.
        - evaporation_rate (float): Rate at which pheromones evaporate.
        - pheromone_deposit (float): Amount of pheromone deposited per solution.
        - alpha (float): Weight of pheromone importance.
        - beta (float): Weight of heuristic importance.
        """
        super().__init__(problem_instance, truck_types=truck_types)
        self.truck_types = list(truck_types)
        self.num_ants = num_ants
        self.generations = config.iterations // num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.alpha = alpha
        self.beta = beta
        self.pheromone_matrix = self.__initialize_pheromone_matrix()

    def search(self) -> 'DeliveryProblem':
        """
        Perform the Ant Colony Optimization search to find the best solution.

        Returns:
        - DeliveryProblem: The best solution found by the algorithm.
        """
        best_solution = None
        best_cost = float('inf')

        for generation in tqdm(range(self.generations), desc="ACO Progress"):
            solutions = []

            # Each ant constructs a solution
            for _ in range(self.num_ants):
                solution = self.__construct_solution()
                cost = self._evaluate_solution(solution)
                solutions.append((solution, cost))

                # Track the best solution
                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost

            # Update pheromones based on solutions
            self.__update_pheromones(solutions)

        return best_solution

    def __initialize_pheromone_matrix(self) -> dict:
        """
        Initialize the pheromone matrix with default values for (order_id, truck_id) pairs.

        Returns:
        - dict: A pheromone matrix initialized with a default value of 1.0.
        """
        return {(order.order_id, truck.truck_id): 1.0 for order in self.problem_instance.orders for truck in self.truck_types}

    def __construct_solution(self) -> 'DeliveryProblem':
        """
        Construct a solution for an ant by assigning orders to trucks and generating valid routes.

        Returns:
        - DeliveryProblem: A valid solution for the current ant.
        """
        solution = copy.deepcopy(self.problem_instance)
        solution.routes = []

        for order in solution.orders:
            # Probabilistically select a truck based on pheromones and heuristics
            probabilities = self.__calculate_truck_probabilities(order)
            truck = random.choices(self.truck_types, weights=probabilities, k=1)[0].copy()

            # Generate a valid route for the truck
            route = self.__generate_valid_route(truck, [order], solution.city_manager)
            if route:
                solution.routes.append(route)

        return solution

    def __generate_valid_route(self, truck, orders, city_manager) -> Route or None:
        """
        Generate a valid route for the truck and orders, ensuring time constraints are met.

        Parameters:
        - truck (Truck): The truck to generate a route for.
        - orders (list): List of orders to deliver.
        - city_manager (CityManager): City manager for calculating distances.

        Returns:
        - Route or None: A valid Route object or None if constraints cannot be met.
        """
        start_city = orders[0].start_city
        end_city = orders[0].end_city

        # Create a list of cities to visit (excluding start and end for now)
        intermediate_cities = [start_city, end_city]
        random.shuffle(intermediate_cities[1:-1])

        # Check if the shuffled route satisfies the time constraints
        if self.__is_route_valid(intermediate_cities, truck, orders, city_manager):
            return Route(truck, orders, city_manager)
        return None

    def __is_route_valid(self, route, truck, orders, city_manager) -> bool:
        """
        Validate a generated route to ensure time constraints are satisfied.

        Parameters:
        - route (list): List of cities in the route.
        - truck (Truck): The truck assigned for the delivery.
        - orders (list): The orders to deliver.
        - city_manager (CityManager): Manages city distances.

        Returns:
        - bool: True if the route is valid, False otherwise.
        """
        total_time = 0
        for i in range(len(route) - 1):
            distance = city_manager.distance_between_cities(city1=route[i], city2=route[i + 1])
            total_time += distance / truck.truck_speed  # Calculate travel time
            # Ensure the total time is within the order deadline
            if total_time > (orders[0].end_time - orders[0].start_time).total_seconds() / 3600:
                return False
        return True

    def __calculate_truck_probabilities(self, order: 'Order') -> list:
        """
        Calculate selection probabilities for trucks based on pheromone and heuristic values.

        Parameters:
        - order (Order): The order to assign.

        Returns:
        - list: Probabilities for selecting each truck.
        """
        probabilities = []
        for truck in self.truck_types:
            pheromone = self.pheromone_matrix.get((order.order_id, truck.truck_id), 1.0)
            heuristic = 1 / (truck.truck_cost + 1e-6)  # Lower cost = better heuristic
            probabilities.append((pheromone ** self.alpha) * (heuristic ** self.beta))

        total = sum(probabilities)
        return [p / total for p in probabilities]

    def __update_pheromones(self, solutions: list):
        """
        Update the pheromone matrix based on the quality of solutions.

        Parameters:
        - solutions (list): A list of tuples (solution, cost).
        """
        # Evaporation step: Reduce all pheromone levels
        for key in self.pheromone_matrix:
            self.pheromone_matrix[key] *= (1 - self.evaporation_rate)

        # Reinforce pheromones based on the quality of solutions
        for solution, cost in solutions:
            pheromone_to_deposit = self.pheromone_deposit / cost
            for route in solution.routes:
                for order in route.orders:
                    key = (order.order_id, route.truck.truck_id)
                    self.pheromone_matrix[key] += pheromone_to_deposit
