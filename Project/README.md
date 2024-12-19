# Route Optimization Problem (ROP)

This project focuses on solving the Route Optimization Problem (ROP) using various optimization algorithms. The goal of this project is to implement and evaluate different algorithms such as Hill Climbing, Genetic Algorithm, Ant Colony Optimization, and Random Search in terms of their efficiency and effectiveness in minimizing the total cost and optimizing truck usage for route planning.

## Project Overview

The Route Optimization Problem (ROP) aims to efficiently plan the best routes for delivering items across various cities with different constraints, such as truck type, weight, and area limits. This project aims to evaluate multiple algorithms to solve this problem using various datasets of small and large order sizes.

### Research Questions

The project answers the following research questions:
1. **How do the optimization algorithms compare in terms of total cost?**
2. **How do the algorithms perform under different problem sizes?**
3. **How does the baseline random search compare with the optimization algorithms?**

### Dataset

The dataset used for this project comes from a publicly available repository on Kaggle. It consists of:
- **Small orders** (up to 50 delivery points)
- **Large orders** (over 200 delivery points)
- **Combined dataset** (small + large)

The dataset includes over 300 cities with varying distances between delivery points, allowing us to evaluate the algorithms’ scalability and performance in realistic route optimization scenarios.

## Installation

To set up the environment for this project, please follow the steps below:

### Step 1: Install Miniconda
1. Download and install Miniconda from [here](https://docs.anaconda.com/miniconda/).
2. Follow the instructions provided in the official Miniconda documentation for your operating system.

### Step 2: Set up the Virtual Environment
1. Create a virtual environment:
    ```bash
    conda create -n 'ROP' python=3.10
    ```
2. Activate the virtual environment:
    ```bash
    conda activate ROP
    ```

### Step 3: Install Dependencies
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

To run the code for the Route Optimization Problem and start evaluating the algorithms, execute the following:

```bash
python main.py
