import logging, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)
sys.path.append(Path)

import config
from models.general import *
from models.prototype import SearchAlgorithm
from models.ant_colony_optimization import AntColonyOptimization
from models.hill_climbing import HillClimbing
from models.random_search import RandomSearch
from models.genetic_algorithm import GeneticAlgorithm
from utility.plot import *


logger = None


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
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
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


def run_algorithm(algorithm_name: str, algorithm_instance: 'SearchAlgorithm', save: bool = True, save_path: str = None) -> dict:
    """
    Run a specified algorithm and return its metrics.
    Should be called within a ProcessPoolExecutor.

    Parameters:
    - algorithm_name (str): The name of the algorithm.
    - algorithm_instance (SearchAlgorithm): The instance of the algorithm to run.
    - save (bool): Whether to save the results to file.
    - save_path (str): The path to save the results.

    Returns:
    - metrics (dict): The metrics of the optimized solution.
    """
    global logger
    if save and save_path is None: raise ValueError("save_path must be provided if save is True.")

    #print(f"\nRunning {algorithm_name}...\n")
    optimized_solution = algorithm_instance.search()
    metrics = optimized_solution.get_metrics()

    if save:
        optimized_solution.save(save_path + f"{algorithm_name.replace(' ', '_')}.pkl")
        optimized_solution.save_to_excel(save_path + f"{algorithm_name.replace(' ', '_')}.xlsx")

    return metrics


def process_algorithm(algorithm_name, algorithm_instance, save: bool = True, save_path: str = None):
    """
    Function to process and run the algorithm in parallel.
    Should be called within a ProcessPoolExecutor.

    Parameters:
    - algorithm_name (str): The name of the algorithm.
    - algorithm_instance (SearchAlgorithm): The instance of the algorithm to run.
    - save (bool): Whether to save the results to file.
    - save_path (str): The path to save the results.

    Returns:
    - algorithm_name (str): The name of the algorithm.
    - _run_algorithm (dict): The metrics of the optimized solution.
    """
    return algorithm_name, run_algorithm(algorithm_name, algorithm_instance, save=save, save_path=save_path)



if __name__ == "__main__":
    loading_path = Path + "Assets/dataset/"
    saving_path = Path + "Assets/output/"

    
    # Setup the logger for debugging
    logger = setup_logger('Main', Path + 'logs/main.log') if config.debug else None
    logger.info('\n----------------------------------------\nStarting the program...\n----------------------------------------\n')
    

    # Create a DataLoader instance
    data_loader = DataLoader(order_small_path = loading_path + "order_small.csv", 
                             order_large_path = loading_path + "order_large.csv", 
                             truck_types_path = loading_path + "truck_types.csv",
                             distance_path = loading_path + "distance.csv")
    

    # Create or Load a DeliveryProblem instance
    if config.load_existing_delivery_problem and os.path.exists(saving_path + "Original.pkl"):
        delivery_problem = pickle.load(open(saving_path + "Original.pkl", "rb")) 
    else:
        delivery_problem = DeliveryProblem(data_loader.orders, data_loader.truck_types, data_loader.city_manager)
    raw_result = delivery_problem.get_metrics()
    logger.info(f"Raw Result: {raw_result}")
    
    delivery_problem.save(saving_path + "Original.pkl")
    delivery_problem.save_to_excel(saving_path + "Original.xlsx")


    # Gather the algorithms to run
    algorithms = {
        "Random Search": RandomSearch(delivery_problem, truck_types=data_loader.truck_types),
        "Hill Climbing": HillClimbing(delivery_problem, truck_types=data_loader.truck_types),
        "Ant Colony Optimization": AntColonyOptimization(
            delivery_problem,
            truck_types=data_loader.truck_types,
            num_ants=50,    # Number of ants in the colony, must be less than config.iterations
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.1,
            pheromone_deposit=1.0
        ),
        "Genetic Algorithm": GeneticAlgorithm(
            delivery_problem,
            truck_types=data_loader.truck_types,
            population_size=100,
            mutation_rate=0.3,
            crossover_rate=0.7
        ),
    }
    results = {}
    

    # Run random search, hill climbing, and ant colony concurrently using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for algorithm_name, algorithm_instance in algorithms.items():
            if algorithm_name != "Genetic Algorithm":   # Skip Genetic Algorithm, run it separately as it requires more resources and time
                futures.append(executor.submit(process_algorithm, algorithm_name, algorithm_instance, save = True, save_path = saving_path))
        
        # Collect metrics
        for future in as_completed(futures):
            algorithm_name, metrics = future.result()
            logger.info(f"{algorithm_name}: {metrics}")
            results[algorithm_name] = metrics


    # Run Genetic Algorithm separately
    ga_name = "Genetic Algorithm"
    metrics = run_algorithm(ga_name, algorithms[ga_name], save=True, save_path=saving_path)
    logger.info(f"{ga_name}: {results}")
    results[ga_name] = metrics
    


    # Plot the results
    results['Initial Solution'] = raw_result  # Add the raw result to the comparison
    draw_overall_comparation(results)
    draw_truck_type_distribution(results)


    logger.info('\n----------------------------------------\nProgram completed successfully.\n----------------------------------------\n')
