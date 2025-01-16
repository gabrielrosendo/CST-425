import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res


x_values = np.linspace(-50, 50, 100)
y_values = sigmoid(x_values)

plt.plot(x_values, y_values)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()