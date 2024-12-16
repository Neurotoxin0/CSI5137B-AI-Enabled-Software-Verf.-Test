import copy, pickle, re, os, sys, uuid
import pandas as pd
from datetime import datetime
from openpyxl.utils import get_column_letter

Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
Path = Path.replace("models/", "")
sys.path.append(Path)

import config


# DataLoder class
class DataLoader:
    def __init__(self, *, order_small_path: str = None, order_large_path: str = None, truck_types_path: str, distance_path: str) -> None:
        """
        Initialize the DataLoader with paths to the dataset files.

        Parameters:
        order_small_path (str): The file path to the small order dataset (CSV).
        order_large_path (str): The file path to the large order dataset (CSV).
        truck_types_path (str): The file path to the truck types dataset (CSV).
        distance_path (str): The file path to the distance matrix dataset (CSV).

        Returns:
        None
        """
        self.order_small_path = order_small_path
        self.order_large_path = order_large_path
        self.truck_types_path = truck_types_path
        self.distance_path = distance_path

        self.city_manager: 'CityManager' = self.__load_distances()
        self.truck_types: list['Truck'] = frozenset(self.__load_trucks())   # list of Truck objects examples that can be created for the problem; frozen set to prevent modifications
        self.orders: list['Order'] = self.__load_orders()

    
    def __load_distances(self) -> 'CityManager':
        """
        Load and parse the distances between cities from the CSV file.

        Parameters:
        None

        Returns:
        Cities: An instance of the Cities class containing the cities and their distances
        """
        city_instance = CityManager()
        distance_df = pd.read_csv(self.distance_path, header=0)
        
        for _, row in distance_df.iterrows():
            city_instance.add_cities_relations(row['Source'], row['Destination'], row['Distance(M)'] * config.distance_scale) # Convert to km

        return city_instance
            
    
    def __load_trucks(self) -> list['Truck']:
        """
        Load and parse the truck types from the CSV file.

        Parameters:
        None

        Returns:
        list: A list of Truck objects.
        """
        trucks = []
        trucks_df = pd.read_csv(self.truck_types_path, header=0)
        
        for _, row in trucks_df.iterrows():
            truck = Truck(
                truck_type=row['Truck Type (length in m)'],
                truck_size=row['Inner Size (m^2)'],
                truck_capacity=row['Weight Capacity (kg)'],
                truck_cost=row['Cost Per KM'],
                truck_speed=row['Speed (km/h)']
            )
            trucks.append(truck)
        
        return trucks
    
    
    def __load_orders(self) -> list['Order']:
        """
        Load and parse the orders from the CSV files.

        Parameters:
        None

        Returns:
        list: A list of Order objects.
        """
        orders = []
        
        # Load small orders
        if self.order_small_path is not None:
            small_orders_df = pd.read_csv(self.order_small_path, header=0)
            for _, row in small_orders_df.iterrows():
                order = Order(
                    order_id=row['Order_ID'],
                    mat_id=row['Material_ID'],
                    item_id=row['Item_ID'],
                    start_city=self.city_manager.get_city(city_name=row['Source'], create=True),
                    end_city=self.city_manager.get_city(city_name=row['Destination'], create=True),
                    start_time=row['Available_Time'],   # Will be converted to datetime object in Order class
                    end_time=row['Deadline'],           # Will be converted to datetime object in Order class
                    danger_type=row['Danger_Type'],
                    area=row['Area'] * config.area_scale,  # Convert to m^2
                    weight=row['Weight'] * config.weight_scale  # Convert to kg
                )
                orders.append(order)

        # Load large orders
        if self.order_large_path is not None:
            large_orders_df = pd.read_csv(self.order_large_path, header=0)
            for _, row in large_orders_df.iterrows():
                order = Order(
                    order_id=row['Order_ID'],
                    mat_id=row['Material_ID'],
                    item_id=row['Item_ID'],
                    start_city=self.city_manager.get_city(city_name=row['Source'], create=True),
                    end_city=self.city_manager.get_city(city_name=row['Destination'], create=True),
                    start_time=row['Available_Time'],
                    end_time=row['Deadline'],
                    danger_type=row['Danger_Type'],
                    area=row['Area'] * config.area_scale,
                    weight=row['Weight'] * config.weight_scale
                )
                orders.append(order)
        
        return orders



# City Manager class
class CityManager:
    # Inner City class
    class City:
        def __init__(self, *, city_id: int = None, city_name: str) -> None:
            """
            Initialize a new city with the given parameters.

            Parameters:
            city_id (int): The unique identifier for the city, if not provided, it will be generated automatically.
            city_name (str): The name of the city.
            
            Returns:
            None
            """
            self.city_id = city_id if city_id is not None else self.__generate_city_id(city_name)
            self.city_name = city_name


        def __generate_city_id(self, city_name: str) -> int:
            """
            Generate a unique city identifier based on the city name.
            If the city name contains a number, return the number, otherwise return a hash of the city name.

            Parameters:
            city_name (str): The name of the city.

            Returns:
            int: The unique city identifier.
            """
            match = re.search(r'\d+', city_name)
            return match.group() if match else hash(city_name)
    

    def __init__(self) -> None:
        """
        Initialize a new city list with the given parameters.

        Returns:
        None
        """
        self.cities = []
        self.distance_matrix = {}  # (city1: 'City', city2: 'City'): distance


    def __str__(self):
        cities_str = ','.join([city.city_name for city in self.cities])  # Adding newline between city names
        return f"Cities: \n{cities_str} \n\nTotal cities: {len(self.cities)}"


    def get_city(self, *, city_id: int = None, city_name: str = None, create: bool = False) -> 'City':
        """
        Get the city ID based on the city name.

        Parameters:
        city_id (int): The unique identifier for the city.
        city_name (str): The name of the city.
        create (bool): Whether to create a new city if it does not exist.

        Returns:
        City: The city object.
        """
        city = None

        if city_id is not None:
            city = next((c for c in self.cities if c.city_id == city_id), None)
        if city is None and city_name is not None:
            city = next((c for c in self.cities if c.city_name == city_name), None)
        if city is None and create:
            status, city = self.add_city(city_id=city_id, city_name=city_name)
            if not status: raise ValueError(f"City {city_name} already exists.")

        return city
    
    
    def add_city(self, *, city_id: int = None, city_name: str = None) -> list[bool, 'City']:
        """
        Add a new city to the list.

        Parameters:
        city_id (int): The unique identifier for the city, if not provided, it will be generated automatically.
        city_name (str): The name of the city.
        
        Returns:
        bool: True if the city was successfully added, False if it already exists.
        City: The city object or None.
        """
        if self.get_city(city_id=city_id, city_name=city_name) is not None: return False, None
        
        city = self.City(city_id=city_id, city_name=city_name)
        self.cities.append(city)
        return True, city

        
    def add_cities_relations(self, city1_name: str, city2_name: str, distance: float) -> None:
        """
        Add a new relation between two cities with the given distance.

        Parameters:
        city1_name (str): The name of the first city.
        city2_name (str): The name of the second city.
        distance (float): The distance between the two cities.

        Returns:
        None
        """
        city1 = self.get_city(city_name=city1_name, create=True)
        city2 = self.get_city(city_name=city2_name, create=True)
        self.distance_matrix[(city1, city2)] = distance
    

    def distance_between_cities(self, *, 
                                city1: 'City' = None, city2: 'City' = None,
                                city1_id: int = None, city2_id: int = None,
                                city1_name: str = None, city2_name: str = None
                                ) -> float:
        """
        Get the distance between two cities.
        Can be called with city objects, IDs, or names.

        Parameters:
        city1 (City): The first city object.
        city2 (City): The second city object.
        
        city1_id (int): The unique identifier for the first city.
        city2_id (int): The unique identifier for the second city.

        city1_name (str): The name of the first city.
        city2_name (str): The name of the second city.

        Returns:
        float: The distance between the two cities.
        """
        if not (city1 and city2):
            city1 = self.get_city(city_id=city1_id, city_name=city1_name)
            city2 = self.get_city(city_id=city2_id, city_name=city2_name)
            if city1 is None or city2 is None:
                raise ValueError(f"City {city1_name} or {city2_name} not found.")
            
        if (city1, city2) not in self.distance_matrix:
            raise ValueError(f"Distance between {city1_name} and {city2_name} is not available.")
        
        return self.distance_matrix[(city1, city2)]
    


# Truck class
class Truck:
    def __init__(self, truck_type: str, truck_size: float, truck_capacity: int, truck_cost: int, truck_speed: float) -> None:
        """
        Initialize a new truck with the given parameters.

        Parameters:
        truck_type (str): The type of the truck.
        truck_size (float): The size of the truck in cubic meters, in m^2.
        truck_capacity (int): The capacity of the truck in kg.
        truck_cost (int): The cost of the truck per km.
        truck_speed (float): The speed of the truck in km/h.

        Returns:
        None
        """
        self.truck_id = self.__generate_truck_id()
        self.truck_type = truck_type
        self.truck_size = eval(truck_size) if isinstance(truck_size, str) else truck_size  # Ensure it's evaluated if it's a string
        self.truck_capacity = truck_capacity
        self.truck_cost = truck_cost
        self.truck_speed = truck_speed
        
        self.current_location = None
        self.cargo = []  # list of orders being carried
        self.remaining_capacity = self.truck_capacity
        self.remaining_area = self.truck_size

    
    def __generate_truck_id(self) -> str:
        """
        Generate a unique truck identifier.

        Parameters:
        None

        Returns:
        str: The unique truck identifier.
        """
        return str(uuid.uuid4())
    

    def copy(self) -> 'Truck':
        """
        Create a copy of the truck object with a new UUID.

        Returns:
        Truck: A new truck object with the same properties but a different UUID.
        """
        # Create a copy of the truck, excluding the truck_id (to generate a new one)
        copied_truck = copy.deepcopy(self)
        copied_truck.truck_id = self.__generate_truck_id()  # Assign a new truck ID
        return copied_truck
        

    
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
            self.remaining_capacity -= order.weight
            self.remaining_area -= order.area
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
            self.remaining_capacity += order.weight
            self.remaining_area += order.area
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
        return (order.weight <= self.remaining_capacity) and (order.area <= self.remaining_area)



# Order class
class Order:
    def __init__(self, order_id: int, mat_id: str, item_id: str, start_city: 'CityManager.City', end_city: 'CityManager.City', start_time: float, end_time: float, danger_type: str, area: float, weight: float) -> None:
        """
        Initialize a new order with the given parameters.

        Parameters:
        order_id (int): The unique identifier for the order.
        mat_id (str): The material identifier.
        item_id (str): The item identifier.
        start_city (str): The name of the starting city.
        end_city (str): The name of the destination city.
        start_time (float): The start time for the delivery.
        end_time (float): The end time for the delivery.
        danger_type (str): The type of danger for the order.
        area (float): The area of the order in cm^2.
        weight (float): The weight of the order in grams.

        Returns:
        None
        """
        self.order_id = order_id
        self.mat_id = mat_id
        self.item_id = item_id
        self.start_city = start_city
        self.end_city = end_city
        self.start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        self.end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        self.danger_type = danger_type
        self.area = area  # in cm^2
        self.weight = weight  # in g
        

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



# Route class
class Route:
    def __init__(self, truck: 'Truck', orders: list['Order'], city_manager: 'CityManager') -> None:
        """
        Initialize a route for the truck with the given orders and cities.

        Parameters:
        truck (Truck): The truck assigned to this route.
        orders (list): A list of Order objects to be delivered.
        cities (Cities): The Cities object containing city data and distances.

        Returns:
        None
        """
        self.truck = truck
        self.orders = orders
        self.city_manager = city_manager

        self.calculate_route_details()

    
    def __clr(self):
        """
        Clear the route details, used when change order of self.orders and recalculating the route.

        Returns:
        None
        """
        self.total_distance = 0
        self.total_cost = 0
        self.route = []


    def calculate_route_details(self) -> None:
        """
        Calculate the details of the route, including total distance, cost, and delivery time.

        Parameters:
        None

        Returns:
        None
        """
        self.__clr()    # Clear the route details before recalculating
        
        # Add the current location of the truck to the route if it's not None
        if self.truck.current_location is not None: self.route.append(self.truck.current_location)  

        # Add distances based on orders and calculate total details
        for order in self.orders:
            # Calculate the distance from the current city to the starting city of the order
            # If the truck is not at any city, assume the truck is in start city
            if self.truck.current_location is not None:
                if self.truck.current_location != order.start_city:
                    distance_to_start = self.city_manager.distance_between_cities(city1=self.truck.current_location, city2=order.start_city)
                    self.total_distance += distance_to_start
                    self.route.append(order.start_city)
                else:   # Truck is already at the start city
                    pass
            
            # Now calculate the distance for the order's route
            distance_to_end = self.city_manager.distance_between_cities(city1=order.start_city, city2=order.end_city)
            self.total_distance += distance_to_end
            #self.truck.current_location = order.end_city  # ONLY CHANGE THE TRUCK LOCATION AFTER DELIVERY IS DONE
            self.route.append(order.end_city)

        # Calculate total cost and delivery time
        self.total_cost = self.truck.truck_cost * self.total_distance


    def __str__(self) -> str:
        coute_str = ', '.join([city.city_name for city in self.route])
        return f"Route: \n{coute_str} \n\nTotal Distance: {self.total_distance} km \n\nTotal Cost: ${self.total_cost} \n\nDelivery Time: {self.delivery_time} hours"



# DeliveryProblem class
class DeliveryProblem:
    def __init__(self, orders: list['Order'], truck_types: list['Truck'], city_manager: 'CityManager') -> None:
        """
        Initialize the delivery problem with the given trucks, orders, and cities.

        Parameters:
        orders (list): A list of Order objects.
        truck_types (list): A list of Truck objects that can be created for the problem.
        cities (Cities): The Cities object containing city data and distances.
        """
        self.orders = orders
        self.truck_types = truck_types
        self.city_manager = city_manager

        self.routes = []

        self.__assign_orders_to_trucks()


    def copy(self) -> 'DeliveryProblem':
        """
        Create a copy of the DeliveryProblem object.

        Returns:
        DeliveryProblem: A new DeliveryProblem object with the same properties.
        """
        return copy.deepcopy(self)
        

    def __assign_orders_to_trucks(self) -> None:
        """
        Assign orders to trucks dynamically, trying to minimize the number of trucks used.
        
        This method will add trucks as needed and try to minimize their usage.

        Parameters:
        None

        Returns:
        None
        """
        for order in self.orders:
            assigned = False
            # First, try to assign this order to an existing truck
            for route in self.routes:
                if route.truck.can_load(order):
                    self.__assign_order_to_truck(order, route=route)
                    assigned = True
                    break

            # If no truck can carry the order, select the appropriate truck type
            if not assigned:
                selected_truck = self.__select_truck_for_order(order, use_dummy=True)
                self.__assign_order_to_truck(order, truck=selected_truck)
    
    
    def __assign_order_to_truck(self,  order: 'Order', *, route: 'Route' = None, truck: 'Truck' = None) -> None:
        """
        Assign the order to a truck and create a route for it.
        Make sure to check if the truck can carry the order before calling this method.

        Parameters:
        order (Order): The order to assign.
        route (Route): The route to add the order to, if not provided, a new route will be created.
        truck (Truck): The truck to assign the order to, no need to provide if route is provided.

        Returns:
        None
        """
        if route is None:
            truck.load_cargo(order)
            route = Route(truck, [order], self.city_manager)
            self.routes.append(route)
        else:
            route.orders.append(order)
            route.truck.load_cargo(order)
            route.calculate_route_details()


    def __select_truck_for_order(self, order: 'Order', *, use_dummy: bool = False) -> 'Truck':
        """
        Select a truck type from the available options to carry the order.
        No optimization is done here, just a simple selection based on the order's weight and area.
        Further algorithms will handle the optimization.

        Parameters:
        order (Order): The order to assign to a truck.
        use_dummy (bool): Whether to use a dummy truck as primary choice -> to verify the optimization algorithm.

        Returns:
        Truck: The selected truck for the order.
        """
        if use_dummy:
            for truck in self.truck_types:
                if truck.truck_type == 'dummy': 
                    if truck.can_load(order):
                        return truck.copy()
                    
        for truck in self.truck_types:
            if truck.can_load(order): return truck.copy()

        # If no truck can carry the order (shouldn't happen ideally), raise an error
        raise ValueError(f"No truck available to carry order {order.order_id}.")


    def __calculate_total_distance(self) -> float:
        """
        Calculate the total distance for all routes.

        Parameters:
        None

        Returns:
        float: The total distance covered by all trucks.
        """
        return sum(route.total_distance for route in self.routes)

    
    def __calculate_total_cost(self) -> float:
        """
        Calculate the total cost for all routes.

        Parameters:
        None

        Returns:
        float: The total cost for all trucks.
        """
        return sum(route.total_cost for route in self.routes)
    

    def __calculate_capacity_utilization(self, truck: 'Truck' = None) -> list[dict]:
        """
        Generate a list of truck types with their respective capacity utilization percentage.

        Parameters:
        truck (Truck): The truck to calculate the utilization for, if not provided, calculate for all trucks.

        Returns:
        list of dicts: Each dictionary contains the truck type and its utilization percentage.
        """
        utilization_data = []
        
        if truck:
            utilization_percentage = ((truck.truck_capacity - truck.remaining_capacity) / truck.truck_capacity) * 100
            utilization_data.append({
                #'Truck ID': truck.truck_id,
                'Truck Type': truck.truck_type,
                'Utilization Percentage': round(utilization_percentage, 2)
            })
        else:
            # Iterate through all trucks and calculate utilization for each
            for route in self.routes:
                truck = route.truck
                utilization_percentage = ((truck.truck_capacity - truck.remaining_capacity) / truck.truck_capacity) * 100
                utilization_data.append({
                    #'Truck ID': truck.truck_id,
                    'Truck Type': truck.truck_type,
                    'Utilization Percentage': round(utilization_percentage, 2)
                })
            
        return utilization_data
    

    def get_metrics(self) -> dict:
        """
        Get the metrics for the delivery problem, including total distance, cost, and number of trucks.

        Parameters:
        None

        Returns:
        dict: A dictionary containing the metrics.
        """
        return {
            'Total Distance': self.__calculate_total_distance(),
            'Total Cost': self.__calculate_total_cost(),
            'Number of Trucks': len(self.routes),
            'Capacity Utilization': self.__calculate_capacity_utilization()
        }
    

    def save(self, filename: str) -> None:
        """
        Save the DeliveryProblem object to a file using pickle.

        Parameters:
        filename (str): The path to the file where the object will be saved.

        Returns:
        None
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file: pickle.dump(self, file)

    
    def save_to_excel(self, filename: str) -> None:
        """
        Save the route assignments, truck details, and orders to an Excel file.

        Parameters:
        filename (str): The path to the file where the data will be saved.

        Returns:
        None
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare the data for the DataFrame
        data = []
        
        for route in self.routes:
            truck = route.truck
            for order in route.orders:
                row = {
                    'Truck_ID': truck.truck_id,
                    'Truck_Route': f"{order.start_city.city_name}->{order.end_city.city_name}",
                    'Order_ID': order.order_id,
                    #'Material_ID': order.mat_id,
                    #'Item_ID': order.item_id,
                    #'Danger_Type': order.danger_type,
                    'Source': order.start_city.city_name,
                    'Destination': order.end_city.city_name,
                    #'Start_Time': order.start_time,
                    #'Arrival_Time': 'N/A', # TODO: Assuming no arrival time for now
                    #'Deadline': order.end_time,
                    #'Shared_Truck': 'N',  # TODO: Assuming no shared trucks for now
                    'Truck Type': truck.truck_type,
                    'Truck Capacity Utilization': self.__calculate_capacity_utilization(truck=truck)[0]['Utilization Percentage']
                }
                data.append(row)

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)

        # Create an ExcelWriter object to save the DataFrame
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

            # Access the workbook and the active sheet
            workbook = writer.book
            sheet = workbook.active

            # Adjust column widths based on the maximum length of the data in each column
            for col in sheet.columns:
                max_length = 0
                column = col[0].column_letter  # Get column name (A, B, C, etc.)
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)  # Add some padding to make it look nicer
                sheet.column_dimensions[column].width = adjusted_width

