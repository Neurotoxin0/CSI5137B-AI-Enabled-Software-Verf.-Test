source of the data: "https://www.kaggle.com/datasets/mexwell/large-scale-route-optimization?resource=download&select=order_small.csv"

- distance.csv记录了所有城市之间的距离，全部保留不需要更改。其中，distance的距离为米（这个是我自己根据数据定的，不然如果是km就跑步过来）

- order_small.csv和order_large.csv中，weight的单位为0.1g（也就是10^-4kg）。这是我自己定义的，因为数据库中最大的weight是31.3million，
为了确保不会出现没有货车能装下的情况，特此设计。

- order_small.csv和order_large.csv中，area的单位是cm^2，以确保所有货物都能装下。

- 在这个project中，有三种卡车，每种卡车数量无限。每种卡车有单位油耗，有装载面积限制，有装载重量限制和速度。一辆车可以装多个订单单的货物。
比如，卡车可以同时装载订单1和订单2的货物，将订单1的货物送到A点，然后再将订单2的货物送到B点。一个订单的货物也可以被分开多个卡车运输，
比如订单3从A点买东西，但一辆货车装不下，那么就会分多辆货车运输。订单有开始时间和结束时间，开始时间之后才能开始运输，并要求在结束时间之前运到
（可以提前，但不能延后）。我们的目的是在保证不会延后的情况下尽可能减少卡车运输的消耗。没有最大停靠点数的限制，且没有停靠费用。