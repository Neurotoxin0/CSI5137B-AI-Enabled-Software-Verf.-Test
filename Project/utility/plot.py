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
    
    # Total cost (in thousands)
    total_costs = [data[algorithm]['Total Cost'] / 1000 for algorithm in algorithms]  # Convert to thousands
    
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
    axes[0].bar(algorithms, total_costs, color=['blue', 'green'], label='Total Cost')
    axes[0].set_title('Total Cost Comparison')
    axes[0].set_ylabel('Cost (Thousands)')
    axes[0].set_xlabel('Algorithms')

    # Label the bars with the actual value
    for i, v in enumerate(total_costs):
        axes[0].text(i, v + 0.1, f"${v:.2f}k", ha='center', va='bottom')  # Adjust positioning if needed

    # Plot 2: Average Truck Utilization Comparison
    avg_utilizations = [np.mean([util['Utilization Percentage'] for util in data[algorithm]['Capacity Utilization']]) for algorithm in algorithms]
    axes[1].bar(algorithms, avg_utilizations, color=['blue', 'green'])
    axes[1].set_title('Truck Utilization Comparison')
    axes[1].set_ylabel('Average Utilization (%)')
    axes[1].set_xlabel('Algorithms')

    # Label the bars with the actual value
    for i, v in enumerate(avg_utilizations):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom')  # Adjust positioning if needed

    # Final adjustment to the layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)  # Adjust space between subplots
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
    num_columns = 2
    num_rows = (num_plots // num_columns) + (1 if num_plots % num_columns != 0 else 0)

    # Create the subplots dynamically
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 6))
    axes = axes.flatten()  # Flatten the axes for easier indexing

    # Plot Truck Type Distribution for each algorithm
    for i, algorithm in enumerate(algorithms):
        truck_type_data = data[algorithm]['Capacity Utilization']  # Extract truck type data for the algorithm
        
        # Aggregating the truck types and their utilization percentages
        truck_type_counts = {}
        total_utilization = sum([util['Utilization Percentage'] for util in truck_type_data])

        # Group the same truck types and sum their utilization percentages
        for util in truck_type_data:
            truck_type = util['Truck Type']
            utilization_percentage = (util['Utilization Percentage'] / total_utilization) * 100
            if truck_type in truck_type_counts:
                truck_type_counts[truck_type] += utilization_percentage
            else:
                truck_type_counts[truck_type] = utilization_percentage

        # Prepare the truck type distribution for pie chart
        truck_types = list(truck_type_counts.keys())
        utilization_percentages = list(truck_type_counts.values())

        # Plot the pie chart for truck type distribution
        axes[i].pie(utilization_percentages, labels=truck_types, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'Truck Type Distribution for {algorithm}')

    # If there are extra axes (empty subplots), hide them
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    # Final adjustment to layout and display
    plt.tight_layout(pad=4.0)  # Adjust the padding between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase space between plots
    plt.savefig('Assets/output/truck_type_comparation.png')



if __name__ == "__main__":
    import pickle, os, sys
    Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
    Path = Path.replace("utility/", "")
    sys.path.append(Path)

    # Load the raw and hillclimbing result data
    raw_data = pickle.load(open(Path + "Assets/output/Original.pkl", "rb"))
    hill_data = pickle.load(open(Path + "Assets/output/Hill_Climbing.pkl", "rb"))
    ant_data = pickle.load(open(Path + "Assets/output/Ant_Colony_Optimization.pkl", "rb"))
    ga_data = pickle.load(open(Path + "Assets/output/Genetic_Algorithm.pkl", "rb"))
    random_data = pickle.load(open(Path + "Assets/output/Random_Search.pkl", "rb"))
    
    # Prepare the data in the required format for drawing comparison
    data = {
        'Initial Solution': raw_data.get_metrics(),
        'Hill Climbing': hill_data.get_metrics(),
        'Ant Colony Optimization' : ant_data.get_metrics(),
        'Genetic Algorithm': ga_data.get_metrics(),
        'Random Search': random_data.get_metrics()
    }

    draw_overall_comparation(data)
    draw_truck_type_distribution(data)
