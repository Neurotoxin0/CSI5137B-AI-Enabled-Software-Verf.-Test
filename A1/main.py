"""
@Description  :   A1 - Main
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import argparse, os, time
import tsp_loader, tsp_solver


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

tsp_instances = []
debug = True


def load_best_known_solutions(filepath) -> dict:
    """
    Load the best known solutions from the given solution file.

    Returns:
    - best_known_solutions (dict): A dictionary mapping TSP file names to the best known solutions.
    """
    best_known_solutions = {}

    try:
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    name = parts[0].strip()
                    best_fitness = float(parts[1].strip().replace(',', ''))  # Remove commas if present
                    best_known_solutions[name] = best_fitness
    except FileNotFoundError:
        return False, f"Error: The file '{filepath}' was not found."
    except Exception as e:
        return False, f"Error reading the file: {str(e)}"
    
    return best_known_solutions


def save_solution_to_csv(tour: list, filename: str = 'solution.csv') -> None:
    """
    Save the tour to a CSV file with a single column of city indices.

    Parameters:
    - tour (list): The list of city indices representing the tour.
    - filename (str): The name of the output CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for city in tour:
            writer.writerow([city])



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load TSP files.')
    parser.add_argument('files', nargs='*', help='Paths to TSP files (optional)')
    args = parser.parse_args()


    # Load TSP files from the command line arguments or prompt the user to choose files
    tsp_loader_instance = tsp_loader.TSPLoader()
    if args.files:
        tsp_loader_instance.load_files_from_args(args.files)
    else:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        tsp_loader_instance.choose_files()
    tsp_instances = tsp_loader_instance.tsp_files
    del tsp_loader_instance

    
    '''
    # Initialize GAOptimizer and find the best parameters
    optimizer = tsp_solver.GAOptimizer(n_iter=20, max_outer_workers=1, max_inner_workers=25)
    best_params = optimizer.optimize(tsp_instances)
    '''
    best_params = {'popsize': 100, 'mutation_rate': 0.05, 'generations': 10, 'tournament_size': 7}   # Manually set based on common practice
    # best_params = {'popsize': 50, 'mutation_rate': 0.14, 'generations': 200, 'tournament_size': 10}   # Found by the optimizer
    
    
    # Generate a genetic algorithm solver for the TSP instances
    ga_instance = tsp_solver.GeneticAlgorithm(
        popsize=best_params['popsize'],
        mutation_rate=best_params['mutation_rate'],
        generations=best_params['generations'],
        tournament_size=best_params['tournament_size']
    )

    
    # Generate a random search algorithm solver for the TSP instances
    random_instance = tsp_solver.RandomSearchAlgorithm(iterations=10)  # make it same as GA for comparison


    # Load the best known solutions for the TSP instances, if given
    best_known_solutions = load_best_known_solutions(Path + 'Assets/tsplib/solutions')


    # Run the solvers on the TSP instances
    for tsp_instance in tsp_instances: 
        # GA solver
        start_time = time.time()
        best_tour, total_cost, validate = ga_instance.solve(tsp_instance)
        tsp_instance.solution = best_tour
        tsp_instance.solver_fitness = total_cost
        tsp_instance.solution_validation = validate
        tsp_duration = time.time() - start_time
        
        # Random solver (baseline)
        if debug:
            best_tour, total_cost = random_instance.solve(tsp_instance)
            tsp_instance.baseline_fitness = total_cost

        # Load the best known solution for the TSP instance, if available
        tsp_instance.best_fitness = best_known_solutions.get(tsp_instance.name, None)

        if debug: print(tsp_instance)

    
    # Print and save according to this assignment's requirement
    if not debug:   
        import csv

        if len(tsp_instances) == 1:
            tsp_instance = tsp_instances[0]
            print(f"{tsp_instance.total_cost:.2f}")
            save_solution_to_csv(tsp_instance.solution, f'solution.csv')
        else:
            for tsp_instance in tsp_instances: 
                print(f"{tsp_instance.name}: {tsp_instance.total_cost:.2f}")
                save_solution_to_csv(tsp_instance.solution, f'{tsp_instance.name}_solution.csv')
    
    # Extensive print and plot for debugging & visualization
    else:
        import matplotlib.pyplot as plt
        import numpy as np
        
        instance_names = [tsp_instance.name for tsp_instance in tsp_instances]
        baseline_fitness = [tsp_instance.baseline_fitness for tsp_instance in tsp_instances]
        solver_fitness = [tsp_instance.solver_fitness for tsp_instance in tsp_instances]
        best_fitness = [tsp_instance.best_fitness for tsp_instance in tsp_instances]
        plt.figure(figsize=(12, 6))
        plt.plot(instance_names, baseline_fitness, 'o-', label='Random Solver (Baseline)', color='blue')
        plt.plot(instance_names, solver_fitness, 'o-', label='GA Solver', color='green')
        plt.plot(instance_names, best_fitness, 'o-', label='Best Known Solution', color='red')

        plt.xticks(rotation=90)  # Rotate instance names for better readability
        plt.xlabel("TSP Problem Instances")
        plt.ylabel("Fitness")
        plt.title("Overview of Fitness for Different TSP Problem Instances")
        plt.legend()

        plt.tight_layout()
        plt.show()
