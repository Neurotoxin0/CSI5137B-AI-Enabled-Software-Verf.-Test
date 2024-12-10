# Truck class
class Truck:
    def __init__(self, truck_id: int, fuel_efficiency: float, load_capacity: float, area_capacity: float, speed: float) -> None:
        """
        Initialize a new truck with the given parameters.

        Parameters:
        truck_id (int): The unique identifier for the truck.
        fuel_efficiency (float): The fuel efficiency of the truck in liters per km.
        load_capacity (float): The maximum weight the truck can carry in kg.
        area_capacity (float): The maximum area the truck can carry in cm^2.
        speed (float): The speed of the truck in km/h.

        Returns:
        None
        """
        self.truck_id = truck_id
        self.fuel_efficiency = fuel_efficiency  # fuel per distance unit
        self.load_capacity = load_capacity  # kg
        self.area_capacity = area_capacity  # cm^2
        self.speed = speed  # km/h
        self.current_location = None
        self.cargo = []  # list of orders being carried

    
    def update_location(self, new_location: 'City') -> None:
        """
        Update the truck's current location.

        Parameters:
        new_location (City): The new city the truck is located at.

        Returns:
        None
        """
        self.current_location = new_location

    
    def load_cargo(self, order: 'Order') -> bool:
        """
        Load cargo into the truck.

        Parameters:
        order (Order): The order to be loaded onto the truck.

        Returns:
        bool: True if the cargo was successfully loaded, False if it exceeds the truck's capacity.
        """
        if self.can_load(order):
            self.cargo.append(order)
            return True
        return False
    

    def unload_cargo(self, order: 'Order') -> bool:
        """
        Unload cargo from the truck.

        Parameters:
        order (Order): The order to be unloaded from the truck.

        Returns:
        bool: True if the cargo was successfully unloaded, False if the order is not in the truck's cargo.
        """
        if order in self.cargo:
            self.cargo.remove(order)
            return True
        return False
    

    def can_load(self, order: 'Order') -> bool:
        """
        Check if the truck can load the given order.

        Parameters:
        order (Order): The order to check for loading feasibility.

        Returns:
        bool: True if the truck can load the order, False otherwise.
        """
        return (order.weight <= self.load_capacity) and (order.area <= self.area_capacity)
    

    def calculate_fuel_consumption(self, distance: float) -> float:
        """
        Calculate the fuel consumption for a given distance.

        Parameters:
        distance (float): The distance to be traveled.

        Returns:
        float: The fuel consumption for the given distance.
        """
        return distance / self.fuel_efficiency



# Order class
class Order:
    def __init__(self, order_id: int, start_location: 'City', end_location: 'City', weight: float, area: float, start_time: float, end_time: float) -> None:
        """
        Initialize a new order with the given parameters.

        Parameters:
        order_id (int): The unique identifier for the order.
        start_location (City): The starting city of the order.
        end_location (City): The ending city of the order.
        weight (float): The weight of the order in 0.1g units.
        area (float): The area of the order in cm^2.
        start_time (float): The time when the order can be delivered.
        end_time (float): The time by which the order must be delivered.

        Returns:
        None
        """
        self.order_id = order_id
        self.start_location = start_location
        self.end_location = end_location
        self.weight = weight  # in 0.1g units
        self.area = area  # in cm^2
        self.start_time = start_time
        self.end_time = end_time


    def is_eligible_for_delivery(self, current_time: float) -> bool:
        """
        Check if the order can be delivered at the current time.

        Parameters:
        current_time (float): The current time to check the eligibility.

        Returns:
        bool: True if the order can be delivered at the current time, False otherwise.
        """
        return self.start_time <= current_time <= self.end_time
    

    def get_delivery_window(self) -> tuple:
        """
        Get the time window for this order's delivery.

        Parameters:
        None

        Returns:
        tuple: The start and end times of the delivery window (start_time, end_time).
        """
        return self.start_time, self.end_time



# City class
class City:
    def __init__(self, city_id: int, name: str, coordinates: tuple) -> None:
        """
        Initialize a new city with the given parameters.

        Parameters:
        city_id (int): The unique identifier for the city.
        name (str): The name of the city.
        coordinates (tuple): The coordinates of the city (latitude, longitude).

        Returns:
        None
        """
        self.city_id = city_id
        self.name = name
        self.coordinates = coordinates  # (latitude, longitude)


    def calculate_distance_to(self, other_city: 'City') -> float:
        """
        Calculate the distance from this city to another using their coordinates.

        Parameters:
        other_city (City): The other city to calculate the distance to.

        Returns:
        float: The distance between the two cities.
        """
        # Assuming Euclidean distance for simplicity (can be replaced with Haversine formula if necessary)
        lat1, lon1 = self.coordinates
        lat2, lon2 = other_city.coordinates
        return ((lat2 - lat1)**2 + (lon2 - lon1)**2) ** 0.5
    

    def get_nearest_city(self, cities: list) -> 'City':
        """
        Find the nearest city from the list of cities.

        Parameters:
        cities (list): A list of City objects to search for the nearest city.

        Returns:
        City: The nearest city object.
        """
        nearest_city = None
        min_distance = float('inf')
        for city in cities:
            distance = self.calculate_distance_to(city)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        return nearest_city



# Route class
class Route:
    def __init__(self, route_id: int, truck: 'Truck', delivery_points: list) -> None:
        """
        Initialize a new route with the given parameters.

        Parameters:
        route_id (int): The unique identifier for the route.
        truck (Truck): The truck assigned to this route.
        delivery_points (list): A list of Order objects representing delivery points.

        Returns:
        None
        """
        self.route_id = route_id
        self.truck = truck
        self.delivery_points = delivery_points  # List of orders to be delivered
        self.total_distance = 0
        self.total_fuel_consumption = 0
        self.delivery_time = 0


    def calculate_route_distance(self) -> float:
        """
        Calculate the total distance for the route.

        Parameters:
        None

        Returns:
        float: The total distance for the route.
        """
        total_distance = 0
        current_location = self.truck.current_location

        for order in self.delivery_points:
            total_distance += current_location.calculate_distance_to(order.start_location)
            current_location = order.end_location
        
        self.total_distance = total_distance
        return total_distance
    

    def calculate_total_fuel(self) -> float:
        """
        Calculate the total fuel consumption for the route.

        Parameters:
        None

        Returns:
        float: The total fuel consumption for the route.
        """
        self.total_fuel_consumption = self.truck.calculate_fuel_consumption(self.total_distance)
        return self.total_fuel_consumption
    

    def calculate_delivery_time(self) -> float:
        """
        Estimate delivery time based on truck speed and route distance.

        Parameters:
        None

        Returns:
        float: The estimated delivery time for the route in hours.
        """
        self.delivery_time = self.total_distance / self.truck.speed
        return self.delivery_time



# DeliveryProblem class
class DeliveryProblem:
    def __init__(self, orders: list, cities: list, trucks: list, distance_matrix: dict) -> None:
        """
        Initialize a new delivery problem with the given parameters.

        Parameters:
        orders (list): A list of Order objects representing all orders in the problem.
        cities (list): A list of City objects representing all cities involved.
        trucks (list): A list of Truck objects representing available trucks.
        distance_matrix (dict): A dictionary representing the distance between cities.

        Returns:
        None
        """
        self.orders = orders  # list of Order objects
        self.cities = cities  # list of City objects
        self.trucks = trucks  # list of Truck objects
        self.distance_matrix = distance_matrix  # matrix of distances between cities


    def add_order(self, order: 'Order') -> None:
        """
        Add a new order to the problem.

        Parameters:
        order (Order): The new order to be added.

        Returns:
        None
        """
        self.orders.append(order)


    def assign_orders_to_trucks(self) -> None:
        """
        Assign orders to trucks based on some criteria (to be implemented).

        Parameters:
        None

        Returns:
        None
        """
        pass


    def generate_distance_matrix(self) -> None:
        """
        Generate the distance matrix for the cities.

        Parameters:
        None

        Returns:
        None
        """
        self.distance_matrix = {}
        for city1 in self.cities:
            for city2 in self.cities:
                if city1 != city2:
                    self.distance_matrix[(city1.city_id, city2.city_id)] = city1.calculate_distance_to(city2)


    def optimize_routes(self) -> None:
        """
        Optimize routes for trucks using search algorithms (to be implemented).

        Parameters:
        None

        Returns:
        None
        """
        pass

