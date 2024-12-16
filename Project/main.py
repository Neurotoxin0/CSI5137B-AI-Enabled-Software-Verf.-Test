import logging, os, sys
from tqdm import tqdm
import matplotlib.pyplot as plt

Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)
sys.path.append(Path)

import config
from models.general import *
from models.hill_climbing import HillClimbing
from models.random_search import RandomSearch
from models.genetic_algorithm import GeneticAlgorithm
from models.ant_colony_optimization import AntColonyOptimization


def setup_logger(logger_name: str, log_file_path: str, *, level=logging.INFO, streamline: bool = False) -> logging.Logger:
    """
    Setup the logger to write to a specified file.
    """
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if streamline:
        cmd_handler = logging.StreamHandler()
        cmd_handler.setFormatter(formatter)
        logger.addHandler(cmd_handler)

    return logger


def run_algorithm(algorithm_name: str, algorithm_instance, save_path: str) -> dict:
    """
    Run a specified algorithm and return its metrics.
    """
    print(f"Running {algorithm_name}...")
    optimized_solution = algorithm_instance.search()
    metrics = optimized_solution.get_metrics()

    # Save results
    optimized_solution.save(save_path + f"{algorithm_name.lower().replace(' ', '_')}.pkl")
    optimized_solution.save_to_excel(save_path + f"{algorithm_name.lower().replace(' ', '_')}.xlsx")

    return metrics


def draw_and_save_comparison_charts(results: dict, raw_result: dict, save_path: str):
    """
    Draw and save comparison charts for total cost and truck utilization.

    Parameters:
    - results (dict): Algorithm results containing metrics.
    - raw_result (dict): Initial raw solution metrics.
    - save_path (str): Directory to save the charts.
    """
    algorithms = list(results.keys())
    total_costs = [results[alg]['Total Cost'] / 1000 for alg in algorithms]  # Convert cost to thousands
    utilizations = [
        sum([truck['Utilization Percentage'] for truck in results[alg]['Capacity Utilization']]) / len(results[alg]['Capacity Utilization'])
        if results[alg]['Capacity Utilization'] else 0
        for alg in algorithms
    ]

    # Add the initial solution to the results
    algorithms.insert(0, 'Initial Solution')
    total_costs.insert(0, raw_result['Total Cost'] / 1000)
    utilizations.insert(0, sum([truck['Utilization Percentage'] for truck in raw_result['Capacity Utilization']]) / len(raw_result['Capacity Utilization']))

    # Create charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Total Cost Comparison
    axes[0].bar(algorithms, total_costs, color=['blue' if alg != 'Random Search' else 'green' for alg in algorithms])
    axes[0].set_title("Total Cost Comparison")
    axes[0].set_ylabel("Cost (Thousands)")
    for i, cost in enumerate(total_costs):
        axes[0].text(i, cost, f"${cost:.2f}k", ha='center', va='bottom', fontsize=9)

    # Truck Utilization Comparison
    axes[1].bar(algorithms, utilizations, color=['blue' if alg != 'Random Search' else 'green' for alg in algorithms])
    axes[1].set_title("Truck Utilization Comparison")
    axes[1].set_ylabel("Average Utilization (%)")
    for i, util in enumerate(utilizations):
        axes[1].text(i, util, f"{util:.1f}%", ha='center', va='bottom', fontsize=9)

    # Adjust layout and save the figures
    for ax in axes:
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=10)

    plt.tight_layout(pad=2.0)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "comparison_charts.png"))  # Save the figure
    print(f"Charts saved to {os.path.join(save_path, 'comparison_charts.png')}")
    plt.show()


if __name__ == "__main__":
    # Setup logger
    logger = setup_logger('Main', Path + 'logs/main.log', streamline=config.debug)

    logger.info('\n----------------------------------------\nStarting the program...\n----------------------------------------\n')

    # Load data
    data_loader = DataLoader(
        order_small_path=Path + "Assets/dataset/order_small.csv",
        truck_types_path=Path + "Assets/dataset/truck_types.csv",
        distance_path=Path + "Assets/dataset/distance.csv"
    )

    # Create DeliveryProblem instance
    delivery_problem = DeliveryProblem(data_loader.orders, data_loader.truck_types, data_loader.city_manager)
    raw_result = delivery_problem.get_metrics()
    logger.info(f"Raw result: {raw_result}")

    # Save initial solution
    delivery_problem.save(Path + "Assets/output/original.pkl")
    delivery_problem.save_to_excel(Path + "Assets/output/original.xlsx")

    # Run all algorithms
    algorithms = {
        "Hill Climbing": HillClimbing(delivery_problem, truck_types=data_loader.truck_types),
        "Random Search": RandomSearch(delivery_problem, truck_types=data_loader.truck_types),
        "Genetic Algorithm": GeneticAlgorithm(
            delivery_problem,
            truck_types=data_loader.truck_types,
            population_size=100,
            mutation_rate=0.3,
            crossover_rate=0.7
        ),
        "Ant Colony Optimization": AntColonyOptimization(
            delivery_problem,
            truck_types=data_loader.truck_types,
            num_ants=50,
            generations=20,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.1,
            pheromone_deposit=1.0
        )
    }

    results = {}
    save_path = Path + "Assets/output/"
    for algorithm_name, algorithm_instance in algorithms.items():
        results[algorithm_name] = run_algorithm(algorithm_name, algorithm_instance, save_path)

    # Draw and save comparison charts
    draw_and_save_comparison_charts(results, raw_result, save_path)

    logger.info('\n----------------------------------------\nProgram completed successfully.\n----------------------------------------\n')
