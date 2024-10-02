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


def print_tsp_instances():
    for tsp_file in tsp_instances:
        print('\n' + '-'*50)
        print(f'File: {tsp_file.filepath}')
        print(f'Name: {tsp_file.name}')
        print(f'Type: {tsp_file.type}')
        print(f'Comment: {tsp_file.comment}')
        print(f'Dimension: {tsp_file.dimension}')
        print(f'Edge Weight Type: {tsp_file.edge_weight_type}')
        print(f'Node Coordinates: {tsp_file.node_coords[:5]} ...')
        print(f'Solution: {tsp_file.solution}')
        print(f'Total Cost: {tsp_file.total_cost}')
        print('\n' + '-'*50)



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

    # Print the loaded TSP instances
    print_tsp_instances()