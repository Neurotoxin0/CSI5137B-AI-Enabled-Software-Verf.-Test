import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


Path = (os.path.split(os.path.realpath(__file__))[0] + "/").replace("\\\\", "/").replace("\\", "/")
os.chdir(Path)


# Creating the data dictionary based on the log details
data = { 
    'Name': [
        'eil51', 'berlin52', 'eil76', 'bier127', 'ch130', 'ch150', 'eil101', 'd198', 'a280', 
        'kroA100', 'gil262', 'kroA150', 'd493', 'fl417', 'rat99', 'fl1577', 'pr226', 'u2152',
        'pcb442', 'kroE100', 'pr299', 'lin318', 'pr107', 'rd100', 'fl3795', 'pr2392', 'pcb3038',
        'rl11849', 'd15112', 'd18512', 'pr124', 'pr136', 'd1291', 'rl1304', 'kroB100', 'rl1323',
        'pr144', 'pr152', 'pr107', 'pr264', 'u574', 'gil262', 'gil317', 'rat783', 'u159'
    ],
    'Dimension': [
        51, 52, 76, 127, 130, 150, 101, 198, 280, 100, 262, 150, 493, 417, 99, 1577, 226, 2152, 
        442, 100, 299, 318, 107, 100, 3795, 2392, 3038, 11849, 15112, 18512, 124, 136, 1291, 1304,
        100, 1323, 144, 152, 107, 264, 574, 262, 317, 783, 159
    ],
    'Duration': [
        70.99, 73.23, 106.78, 179.24, 184.48, 212.31, 141.25, 284.27, 409.90, 141.48, 385.32,
        208.74, 764.40, 633.57, 133.21, 3006.50, 312.72, 4680.22, 646.03, 136.77, 424.77, 
        453.24, 139.93, 136.67, 9970.72, 5205.20, 7254.78, 58444.41, 90192.25, 126711.39, 
        167.97, 182.36, 2347.76, 2344.08, 138.21, 2374.26, 193.18, 204.14, 139.93, 369.09,
        877.32, 385.32, 651.91, 1264.53, 215.48
    ]
}
df = pd.DataFrame(data)

# Calculate correlation coefficient between Dimension and Duration
correlation = np.corrcoef(df['Dimension'], df['Duration'])[0, 1]

# Plot the data to show the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['Dimension'], df['Duration'], color='blue', alpha=0.7)
plt.plot(df['Dimension'], np.poly1d(np.polyfit(df['Dimension'], df['Duration'], 1))(df['Dimension']), color='red', label=f'Linear fit (correlation: {correlation:.2f})')

plt.title('Relation between Problem Dimension and Duration')
plt.xlabel('Dimension (Number of Cities)')
plt.ylabel('Duration (seconds)')
plt.grid(True)
plt.legend()
plt.tight_layout()

#plt.show()
plt.savefig('runtime_graph.png')
