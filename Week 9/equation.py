import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, n):
    x_values = [x0]
    for _ in range(n - 1):
        x_values.append(r * x_values[-1] * (1 - x_values[-1]))
    return x_values

r = 3.2      # Driving parameter
x0 = 0.25    # Initial population
n = 100      # Number of iterations

population = logistic_map(r, x0, n)

# Plot results
plt.plot(range(n), population, marker='o', linestyle='-', markersize=2)
plt.xlabel('Iteration')
plt.ylabel('Population')
plt.title(f'Logistic Map (r={r}, x0={x0})')
plt.show()
