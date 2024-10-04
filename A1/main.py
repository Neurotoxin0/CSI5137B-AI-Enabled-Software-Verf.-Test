"""
@Description  :   A1 - Main
@Author1      :   Yang Xu, 300342009
@Author2      :   Peizhou Zhang, 300400642
@Comment      :   Dev with Python 3.10.0
"""

import argparse, os
import tsp_loader, tsp_solver


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)

tsp_instances = []



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load TSP files.')
    parser.add_argument('files', nargs='*', help='Paths to TSP files (optional)')
    args = parser.parse_args()


    # Load TSP files from the command line arguments or prompt the user to choose files
    tsp_loader = tsp_loader.TSPLoader()
    if args.files:
        tsp_loader.load_files_from_args(args.files)
    else:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        tsp_loader.choose_files()
    tsp_instances = tsp_loader.tsp_files


    # Generate a genetic algorithm solver for the TSP instances
    ga = tsp_solver.GeneticAlgorithm(popsize=50, mutation_rate=0.1, generations=100)
    for tsp_instance in tsp_instances: 
        print(f'Running GA solver on {tsp_instance.name} with {tsp_instance.dimension} cities...')
        best_tour, total_cost = ga.solve(tsp_instance.node_coords)
        tsp_instance.solution = best_tour
        tsp_instance.total_cost = total_cost


    # Print the loaded TSP instances
    for tsp_instance in tsp_instances: print(tsp_instance)