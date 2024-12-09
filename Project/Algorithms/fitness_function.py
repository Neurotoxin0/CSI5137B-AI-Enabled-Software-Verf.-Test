class FitnessFunction:
    '''
    How to use:
    fitness_function = FitnessFunction(truck_types=[
        {'fuel_rate': 0.5, 'weight_limit': 1000, 'area_limit': 50},
        {'fuel_rate': 0.8, 'weight_limit': 1500, 'area_limit': 70},
        {'fuel_rate': 1.2, 'weight_limit': 2000, 'area_limit': 100}
    ])
    '''
    def __init__(self, truck_types):
        """
        Initialize the FitnessFunction with truck types and default penalties.
        
        :param truck_types: List of dictionaries containing truck attributes (e.g., fuel_rate, weight_limit, area_limit).
        """
        self.truck_types = truck_types
        # Define penalties directly within the fitness function
        self.penalties = {
            'time': 10000,  # Penalty for each delayed order
            'load': 10000    # Penalty for each overloaded truck
        }

    def calculate(self, solution):
        """
        Calculate the fitness value for a given solution.
        
        :param solution: Object containing the solution details, including trucks and orders.
        :return: Fitness value (lower is better).
        """
        total_cost = 0
        total_penalty = 0
        
        for truck in solution.trucks:
            # Calculate fuel cost
            total_cost += truck.fuel_rate * truck.total_distance
            
            # Check load constraints
            if truck.total_weight > truck.weight_limit or truck.total_area > truck.area_limit:
                total_penalty += self.penalties['load']
            
            # Check time constraints
            for order in truck.orders:
                if order.delivery_time > order.deadline:
                    total_penalty += self.penalties['time']
        
        # Total fitness = total cost + total penalty
        return total_cost + total_penalty
