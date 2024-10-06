"""
@Description  :   A1 - Main
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import argparse, csv, os
import tsp_loader, tsp_solver


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

tsp_instances = []
debug = True


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


    # Initialize GAOptimizer and find the best parameters
    optimizer = tsp_solver.GAOptimizer(n_iter=20, max_outer_workers=2, max_inner_workers=10)
    best_params = optimizer.optimize(tsp_instances)
    
    '''
    # best_params = {'popsize': 100, 'mutation_rate': 0.1, 'generations': 100, 'tournament_size': 5}
    # Generate a genetic algorithm solver for the TSP instances
    ga_instance = tsp_solver.GeneticAlgorithm(
        popsize=best_params['popsize'],
        mutation_rate=best_params['mutation_rate'],
        generations=best_params['generations'],
        tournament_size=best_params['tournament_size']
    )
    for tsp_instance in tsp_instances: 
        if debug: print(f'Running GA solver on {tsp_instance.name} with {tsp_instance.dimension} cities...')
        best_tour, total_cost, validate = ga_instance.solve(tsp_instance.node_coords)
        tsp_instance.solution = best_tour
        tsp_instance.total_cost = total_cost
        tsp_instance.solution_validation = validate

    
    # Print summary
    if debug:
        tsp_scorer_instance = tsp_loader.TSPScorer(Path + 'Assets/tsplib/solutions')

        for tsp_instance in tsp_instances: 
            print(tsp_instance)
            print(f"Ratio: {tsp_scorer_instance.validate_fitness(tsp_instance):.2f}")
    else:   # print and save according to this assignment's requirement
        if len(tsp_instances) == 1:
            tsp_instance = tsp_instances[0]
            print(f"{tsp_instance.total_cost:.2f}")
            save_solution_to_csv(tsp_instance.solution, f'solution.csv')
        else:
            for tsp_instance in tsp_instances: 
                print(f"{tsp_instance.name}: {tsp_instance.total_cost:.2f}")
                save_solution_to_csv(tsp_instance.solution, f'{tsp_instance.name}_solution.csv')
    '''