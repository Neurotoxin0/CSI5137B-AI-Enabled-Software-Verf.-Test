"""
@Description  :   Run Random Search for TSP independently with CSV output
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import random
import math
import time
import csv
import tsp_loader


def euclidean_distance(city1: tuple, city2: tuple) -> float:
    """
    Calculate the Euclidean distance between two cities.

    Parameters:
    - city1 (tuple): A tuple containing the x and y coordinates of the first city.
    - city2 (tuple): A tuple containing the x and y coordinates of the second city.

    Returns:
    - distance (float): The Euclidean distance between the two cities.
    """
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def total_distance(tour: list, cities: list) -> float:
    """
    Calculate the total distance of a tour by summing the distances between consecutive cities.

    Parameters:
    - tour (list): A list of city indices representing the tour.
    - cities (list): A list of tuples containing city indices and their coordinates.

    Returns:
    - distance (float): The total distance of the tour.
    """
    distance = 0
    for i in range(len(tour)):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % len(tour)]]  # Connect the last city back to the first
        distance += euclidean_distance(city1, city2)
    return distance


class RandomSearchSolver:
    """
    A class to represent a random search solver for the Travelling Salesman Problem (TSP).

    Attributes:
    - cities (list): The list of cities with their coordinates.
    - best_tour (list): The best tour found so far.
    - best_fitness (float): The total distance of the best tour.
    - iterations (int): The number of iterations to perform random search.
    
    Methods:
    - solve(): Run the random search and return the best tour and its total distance.
    """

    def __init__(self, *, iterations: int) -> None:
        """
        Initialize the random search solver with the number of iterations.

        Parameters:
        - iterations (int): The number of iterations to run the random search.
        """
        self.cities = []
        self.best_tour = None
        self.best_fitness = float('inf')
        self.iterations = iterations

    def solve(self, node_coords: list) -> list:
        """
        Run the random search to find the best solution.

        Parameters:
        - node_coords (list): A list of tuples containing the city indices and coordinates.

        Returns:
        - list: A list of total distances (fitness) for each iteration.
        """
        self.cities = node_coords
        num_cities = len(self.cities)

        distances = []
        for i in range(self.iterations):
            # Generate a random tour
            tour = random.sample(range(num_cities), num_cities)
            fitness = total_distance(tour, self.cities)

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_tour = tour

            distances.append(self.best_fitness)

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Random Search Iteration {i + 1}: Best Distance = {self.best_fitness:.2f}")
        
        return distances


def save_random_search_results_to_csv(distances, filename='random_search_result.csv'):
    """
    Save the results of Random Search to a CSV file.

    Parameters:
    - distances (list): List of distances at each iteration.
    - filename (str): The name of the output CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Best Distance'])

        for i, distance in enumerate(distances):
            writer.writerow([i + 1, distance])
    
    print(f"Random Search results saved to {filename}")


if __name__ == '__main__':
    # Load TSP file using your existing TSP loader
    tsp_loader_instance = tsp_loader.TSPLoader()
    
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    tsp_loader_instance.choose_files()
    tsp_instances = tsp_loader_instance.tsp_files

    # Number of iterations for random search
    random_search_iterations = 500

    for tsp_instance in tsp_instances:
        print(f"Running Random Search on TSP instance {tsp_instance.name} with {tsp_instance.dimension} cities...")

        # Run random search
        random_solver = RandomSearchSolver(iterations=random_search_iterations)
        distances = random_solver.solve(tsp_instance.node_coords)

        # Save the results to CSV
        save_random_search_results_to_csv(distances)

        # Print the best solution found
        print(f"Best distance found: {random_solver.best_fitness:.2f}")
