"""
@Description  :   A1 - Main
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import argparse, logging, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import tsp_loader, tsp_solver


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

tsp_instances = []
debug = False
iterations = 500     # 500, as recommended by best common practice
max_workers = 20


if debug:
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
else:
    import csv


def setup_logger(logger_name: str, log_file_path: str, *, level = logging.INFO, streamline: bool = False) -> logging.Logger:
    """
    Setup the logger to write to a specified file.

    Parameters:
    - logger_name (str): The name of the logger.
    - log_file_path (str): The path to the log file.
    - level (int): The logging level.
    - streamline (bool): Whether to output the log to the CLI interface.
    
    Returns:
    - logger (logging.Logger): The logger object.
    """

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create and add file handler
    if not os.path.exists(os.path.dirname(log_file_path)): os.makedirs(os.path.dirname(log_file_path))
    file_handler = logging.FileHandler(log_file_path)
    #file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create a stream handler
    if streamline:
        cmd_handler = logging.StreamHandler()
        #cmd_handler.setLevel(level)
        cmd_handler.setFormatter(formatter)
        logger.addHandler(cmd_handler)
    
    return logger


def load_best_known_solutions(filepath) -> dict:
    """
    Load the best known solutions from the given solution file.

    Parameters:
    - filepath (str): The path to the solution file.

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


def solve(tsp_instance, ga_instance, random_instance, *, progress_bar_index: int) -> tuple:
    """
    Helper function to run both GA solver and Random Search on a single TSP instance.

    Parameters:
    - tsp_instance (TSPFile): The TSP instance to solve.
    - ga_instance (GeneticAlgorithm): The GA solver instance.
    - random_instance (RandomSearchAlgorithm): The random search solver instance.
    - progress_bar_index (int): The index of the progress bar.

    Returns:
    - tsp_instance (TSPFile): The updated TSP instance with the solution and fitness values.
    - ga_cost_list (list): The list of fitness values for the GA solver.
    - rd_cost_list (list): The list of fitness values for the random solver.
    """
    # GA solver
    start_time = time.time()
    best_tour, validate, ga_total_cost, ga_cost_list = ga_instance.solve(tsp_instance, progress_bar_index=progress_bar_index)
    tsp_instance.solution = best_tour
    tsp_instance.solver_fitness = ga_total_cost
    tsp_instance.solution_validation = validate
    tsp_instance.duration = time.time() - start_time
    
    # Random solver (baseline)
    rd_cost_list = []
    if debug:
        best_tour, rd_total_cost, rd_cost_list = random_instance.solve(tsp_instance)
        tsp_instance.baseline_fitness = rd_total_cost
    
    return tsp_instance, ga_cost_list, rd_cost_list


def draw_overview_plot() -> None:
    """
    Draw an overview plot of the fitness for different TSP problem instances.
    """
    instance_names = [tsp_instance.name for tsp_instance in tsp_instances]
    baseline_fitness = [tsp_instance.baseline_fitness for tsp_instance in tsp_instances]
    solver_fitness = [tsp_instance.solver_fitness for tsp_instance in tsp_instances]
    #best_fitness = [tsp_instance.best_fitness for tsp_instance in tsp_instances]
    
    plt.figure(figsize=(12, 6))
    plt.plot(instance_names, baseline_fitness, 'o-', label='Random Solver (Baseline)', color='blue')
    plt.plot(instance_names, solver_fitness, 'o-', label='GA Solver', color='green')
    #plt.plot(instance_names, best_fitness, 'o-', label='Best Known Solution', color='red')

    plt.xticks(rotation=90)  # Rotate instance names for better readability
    plt.xlabel("TSP Problem Instances")
    plt.ylabel("Fitness")
    plt.title("Overview Fitness Comparison for Different TSP Problems")
    plt.legend()

    plt.tight_layout()
    #plt.show()
    if not os.path.exists('Assets/plots'): os.makedirs('Assets/plots')
    plt.savefig('Assets/plots/overall_fitness_comparison.png')


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
    # Setup the logger for debugging
    logger = setup_logger('Main', Path + 'logs/main.log') if debug else None

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

    
    # Load the best known solutions for the TSP instances, if given
    best_known_solutions = load_best_known_solutions(Path + 'Assets/tsplib/solutions')
    
    
    # Prameter setting
    '''
    # Initialize GAOptimizer and find the best parameters
    optimizer = tsp_solver.GAOptimizer(n_iter=20, max_outer_workers=1, max_inner_workers=25)
    best_params = optimizer.optimize(tsp_instances)
    '''
    best_params = {'popsize': 100, 'mutation_rate': 0.05, 'generations': iterations, 'tournament_size': 7}   # Manually set based on common practice
    # best_params = {'popsize': 50, 'mutation_rate': 0.14, 'generations': 200, 'tournament_size': 10}   # Found by the optimizer
    

    # Create solver instances
    ga_instance = tsp_solver.GeneticAlgorithm(
        popsize=best_params['popsize'],
        mutation_rate=best_params['mutation_rate'],
        generations=best_params['generations'],
        tournament_size=best_params['tournament_size']
    )
    random_instance = tsp_solver.RandomSearchAlgorithm(iterations=iterations)  # make its iterations same as GA for comparison

    # Run solvers concurrently using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        main_progress_bar = tqdm(total=len(tsp_instances), desc="Solving TSP Instances", position=0, leave=True) if debug else None
        futures = []
            
        for idx, tsp_instance in enumerate(tsp_instances):
            # Submit the task to the executor, with different progress bar index for each task
            futures.append(executor.submit(solve, tsp_instance, ga_instance, random_instance, progress_bar_index=idx+1))
        
        # Collect results
        for future in as_completed(futures):
            updated_tsp_instance, ga_cost_list, rd_cost_list = future.result()

            # Load the best known solution for the TSP instance, if available
            updated_tsp_instance.best_fitness = best_known_solutions.get(updated_tsp_instance.name, None)

            # Update the corresponding tsp_instance in tsp_instances
            for idx, instance in enumerate(tsp_instances):
                if instance.name == updated_tsp_instance.name:
                    tsp_instances[idx] = updated_tsp_instance
                    break

            # Print and store the visualized result (detailed fitness comparison)
            if debug: 
                main_progress_bar.update(1)

                logger.info(updated_tsp_instance)
                iteration_range = list(range(1, iterations + 1))

                plt.figure(figsize=(10, 5))
                plt.plot(iteration_range, ga_cost_list, 'o-', label='GA Solver', color='green')
                if len(rd_cost_list) > 0: plt.plot(iteration_range, rd_cost_list, 'o-', label='Random/Baseline', color='blue')
                
                plt.xlabel("Iterations")
                plt.ylabel("Fitness")
                plt.title(f"Detailed Fitness Comparison for `{updated_tsp_instance.name}.tsp`")
                plt.legend()

                plt.tight_layout()
                #plt.show()
                if not os.path.exists('Assets/plots'): os.makedirs('Assets/plots')
                plt.savefig(f'Assets/plots/{updated_tsp_instance.name}_fitness_comparison.png')

                # Draw / Update the overview plot for all TSP instances
                draw_overview_plot()   
        if debug: main_progress_bar.close() 

    
    # Print and save according to this assignment's requirement
    if not debug:   
        if len(tsp_instances) == 1:
            tsp_instance = tsp_instances[0]
            print(f"{tsp_instance.solver_fitness:.2f}")
            save_solution_to_csv(tsp_instance.solution, f'solution.csv')
        else:
            for tsp_instance in tsp_instances: 
                print(f"{tsp_instance.name}: {tsp_instance.solver_fitness:.2f}")
                save_solution_to_csv(tsp_instance.solution, f'{tsp_instance.name}_solution.csv')