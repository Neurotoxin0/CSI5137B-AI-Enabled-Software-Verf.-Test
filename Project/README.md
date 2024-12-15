# TODO
+ order time constraint
+ partial load

# Limitatiom
+ assume order is picked up at start city and directly go to end city, then process next order, best practice is to load orders and deliver orders simultaneously 
    + Update: now multiple orders can be carried simultaneously, but the cost for pickup the order is not calculated (truck now can pick up orders but dont need be in that city)
+ did not consider order delivery windows (start / deadline)
+ did not consider partial load, meaning order can either be loaded with full amount or not load at all
+ mention why there is no graph for # of trucks comparation -> algorithm dev limitation -> lack of methods to merge trucks