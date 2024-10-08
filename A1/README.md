# TSP Solver with Genetic Algorithm Optimizer

## Overview

This project provides a tool to solve the Travelling Salesman Problem (TSP) using a Genetic Algorithm (GA). It includes functionalities to load native TSP file(s), run the genetic algorithm, and output the solution(s). 

The project also includes an optimizer to fine-tune GA parameters (turned off and use pre-set hyper-parameters by default); as well as a scorer to score the solution based on input `best solution` file.

## Structure

1. **main.py**: The entry point to load TSP file(s), optionally optimize parameters, solve and output instance(s).
2. **tsp_loader.py**: Handles loading and validation of TSP file(s); Validates the solution against known best solutions and provides a score ratio.
3. **tsp_solver.py**: Implements the genetic algorithm and parameter optimizer.

## How to Run

### Prerequisites

- Python 3.10.0 or later.
- Coded & Tested under Windows system.
- Required internal libraries: `argparse`, `csv`, `os`, `tkinter`, `concurrent.futures`.
- Required external libraries: `None`.

### Steps

1. **Via Command Line**:
   - Run the script with TSP file path(s) as argument(s):
     ```bash
     python main.py file1.tsp file2.tsp ...
     ```
   
2. **Via File Dialog**:
   - If no file paths are provided, a file dialog will prompt you to select TSP file(s).

3. **Output**:
   - Displays the best tour cost for each given TSP problem.
   - Saves solutions in CSV file(s) named `solution.csv` if only a single TSP problem, or `<instance_name>_solution.csv` for multiple instances.

## Key Features

- **Command Line Arguments**: Supports loading TSP file(s) through command line or file dialog.
- **Genetic Algorithm**: Uses `tsp_solver.py` to run GA with parameters like population size, mutation rate, generations, and tournament size.
- **Hyperparameter Optimization**: Optionally optimizes GA parameters using the `GAOptimizer` class in `tsp_solver.py`.
- **Solution Scorer**: Validates the solution against known best solutions using `tsp_loader.py` and provides a ratio to measure performance.
- **Solution Export**: Saves the best-found tour to a CSV file for each TSP instance.

## Logging

- The optimizer logs its progress to `ga_optimizer.log`.

## Example

- To run the script for the TSP instance `berlin52.tsp`:
    ```bash
    python main.py berlin52.tsp
    ```

- To use the hyperparameter optimizer: uncomment corresponding session in `main.py` -> `if __name__ == '__main__'` block 

- To output full results (print only) as well as scoring ratio: change `main.py` -> `debug` to `True`; rename and put solution file as `Assets/tsplib/solutions`

## Default Params & Run Time

- `best_params = {'popsize': 100, 'mutation_rate': 0.05, 'generations': 500, 'tournament_size': 7}`   
   - Manually set based on common practice.
   - Estimated Run Time: `303.37` seconds (based on the `a280.tsp` file, which has 280 nodes (DIMENSION)).
