import matplotlib.pyplot as plt
import numpy as np

def draw_overall_comparation(data: dict):
    """
    Draws a comparison of the overall performance based on total cost and truck utilization.
    
    Parameters:
    data (dict): A dictionary containing solving algorithms (e.g., 'hillclimb', 'raw_result') as keys
                    and their respective result dictionaries as values.
    
    Returns:
    None
    """
    algorithms = list(data.keys())
    
    # Total cost
    total_costs = [data[algorithm]['Total Cost'] for algorithm in algorithms]
    
    # Number of trucks used
    num_trucks_used = [len(data[algorithm]['Capacity Utilization']) for algorithm in algorithms]

    # Truck type distribution (percentage)
    truck_type_data = {algorithm: [] for algorithm in algorithms}  # store truck type percentages
    for algorithm in algorithms:
        capacity_utilization = data[algorithm]['Capacity Utilization']
        
        # Calculate total utilization for normalization
        total_util = sum(util['Utilization Percentage'] for util in capacity_utilization)
        
        # Store each truck type's utilization percentage in the truck_type_data
        for util in capacity_utilization:
            truck_type_data[algorithm].append(
                (util['Truck Type'], (util['Utilization Percentage'] / total_util) * 100)
            )

    
    # Plotting the results
    fig, axes = plt.subplots(1, 2)

    # Plot 1 : Total Cost Comparison
    axes[0,].bar(algorithms, total_costs, color=['blue', 'green'], label='Total Cost')
    axes[0].set_title('Total Cost Comparison')
    axes[0].set_ylabel('Cost')
    axes[0].set_xlabel('Algorithms')

    # Plot 2: Average Truck Utilization Comparison
    avg_utilizations = [np.mean([util['Utilization Percentage'] for util in data[algorithm]['Capacity Utilization']]) for algorithm in algorithms]
    axes[1].bar(algorithms, avg_utilizations, color=['blue', 'green'])
    axes[1].set_title('Average Truck Utilization Comparison')
    axes[1].set_ylabel('Average Utilization (%)')
    axes[1].set_xlabel('Algorithms')

    # Final adjustment to the layout
    plt.tight_layout()
    plt.savefig('Assets/output/overall_comparison.png')



def draw_truck_type_distribution(data):
    """
    Draw the truck type distribution pie chart for each algorithm in the given data.

    Parameters:
    data (dict): Dictionary containing the results for each algorithm, with truck type distribution information.

    Returns:
    None
    """
    algorithms = list(data.keys())  # Extract algorithm names
    num_plots = len(algorithms)  # Number of algorithms to plot

    # Dynamically calculate the number of rows and columns for the subplots
    num_columns = 2  # Set the number of columns (you can adjust this as needed)
    num_rows = (num_plots // num_columns) + (1 if num_plots % num_columns != 0 else 0)  # Calculate rows needed

    # Create the subplots dynamically
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 6))
    axes = axes.flatten()  # Flatten the axes for easier indexing

    # Plot Truck Type Distribution for each algorithm
    for i, algorithm in enumerate(algorithms):
        truck_type_data = data[algorithm]['Capacity Utilization']  # Extract truck type data for the algorithm
        truck_types = [val['Truck Type'] for val in truck_type_data]
        utilization_percentages = [val['Utilization Percentage'] for val in truck_type_data]

        axes[i].pie(utilization_percentages, labels=truck_types, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'Truck Type Distribution for {algorithm}')

    # If there are extra axes (empty subplots), hide them
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    # Final adjustment to layout and display
    plt.tight_layout()
    plt.savefig('Assets/output/truck_type_comparation.png')



if __name__ == "__main__":
    import pickle, os, sys
    Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
    Path = Path.replace("utility/", "")
    sys.path.append(Path)

    # Load the raw and hillclimbing result data
    raw_data = pickle.load(open(Path + "Assets/output/original.pkl", "rb"))
    hill_data = pickle.load(open(Path + "Assets/output/hill_climbing.pkl", "rb"))
    
    # Prepare the data in the required format for drawing comparison
    data = {
        'Initial Solution': raw_data.get_metrics(),
        'Hill Climbing': hill_data.get_metrics()
    }

    #draw_overall_comparation(data)
    draw_truck_type_distribution(data)
