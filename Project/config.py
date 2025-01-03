# --------------------------------------------------
debug = True
iterations = 1000
max_workers = 25     # Maximum number of workers for concurrent processing
load_existing_delivery_problem = True   # Load existing DeliveryProblem instance from file if True, else create a new instance

distance_scale = 0.001   # Convert distance (distance.csv) from meters to kilometers
weight_scale = 0.0001    # Convert weight (order_small.csv / order_large.csv) from 0.1 grams to kilograms
area_scale = 0.0001      # Convert area (order_small.csv / order_large.csv) from cm^2 to m^2 
# --------------------------------------------------
