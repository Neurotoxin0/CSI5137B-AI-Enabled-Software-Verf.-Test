import logging, os, sys
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)
sys.path.append(Path)

import config
from models.general import *
from models.hill_climbing import HillClimbing


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



if __name__ == "__main__":
    # Setup the logger for debugging
    logger = setup_logger('Main', Path + 'logs/main.log') if config.debug else None
    logger.info('\n----------------------------------------\nStarting the program...\n----------------------------------------\n')
    

    # Create a DataLoader instance
    data_loader = DataLoader(order_small_path = Path + "Assets/dataset/order_small.csv", 
                             #order_large_path = Path + "Assets/dataset/order_large.csv", 
                             truck_types_path = Path + "Assets/dataset/truck_types.csv",
                             distance_path = Path + "Assets/dataset/distance.csv")
    

    # Create a DeliveryProblem instance
    delivery_problem = DeliveryProblem(data_loader.orders, data_loader.truck_types, data_loader.city_manager)
    raw_result = delivery_problem.get_metrics()
    logger.info(f"Raw result: {raw_result}")
    #delivery_problem.save_to_excel(Path + "Assets/output/originl.xlsx")


    # Initialize the HillClimbing algorithm with the problem instance
    hill_climbing = HillClimbing(delivery_problem, truck_types=data_loader.truck_types)
    hill_optimized = hill_climbing.search()
    hill_result = hill_optimized.get_metrics()
    logger.info(f"Hill climbing result: {hill_result}")


    input("Press Enter to continue...")


    '''data_loader.city_manager.get_city(city_id=1212)
    data_loader.city_manager.get_city(city_name='namename')

    data_loader.city_manager.distance_between_cities(city1=, city2=)
    data_loader.city_manager.distance_between_cities(city1_id=123, city2_id=456)
    data_loader.city_manager.distance_between_cities(city1_name='city1', city2_name='city2')

    data_loader.truck_types[0].can_load(order)'''





