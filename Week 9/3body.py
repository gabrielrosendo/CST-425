import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Gravitational constant
G = 1  

def three_body_equations(t, state, m1, m2, m3):
    """Compute derivatives for the three-body problem."""
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state

    # Distances between bodies
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Compute acceleration using Newton's law
    ax1 = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    ay1 = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3
    ax2 = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    ay2 = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3
    ax3 = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    ay3 = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# Initial conditions (positions and velocities)
m1, m2, m3 = 1.0, 1.0, 1.0  # Masses of bodies
state0 = [
    -1, 0,  # x1, y1
    1, 0,   # x2, y2
    0, 1,   # x3, y3
    0.5, 0.5,  # vx1, vy1
    -0.5, -0.5, # vx2, vy2
    0, 0      # vx3, vy3
]

# Time span for simulation
t_span = (0, 10)
t_eval = np.linspace(*t_span, 500)

# Solve the equations
sol = solve_ivp(three_body_equations, t_span, state0, t_eval=t_eval, args=(m1, m2, m3))

# Extract positions
x1, y1 = sol.y[0], sol.y[1]
x2, y2 = sol.y[2], sol.y[3]
x3, y3 = sol.y[4], sol.y[5]

# Animation
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Three-Body Problem")

# Traces and points
trace1, = ax.plot([], [], 'r-', alpha=0.6)
trace2, = ax.plot([], [], 'g-', alpha=0.6)
trace3, = ax.plot([], [], 'b-', alpha=0.6)
point1, = ax.plot([], [], 'ro', markersize=6)
point2, = ax.plot([], [], 'go', markersize=6)
point3, = ax.plot([], [], 'bo', markersize=6)

def update(frame):
    """Update function for animation."""
    trace1.set_data(x1[:frame], y1[:frame])
    trace2.set_data(x2[:frame], y2[:frame])
    trace3.set_data(x3[:frame], y3[:frame])
    point1.set_data(x1[frame], y1[frame])
    point2.set_data(x2[frame], y2[frame])
    point3.set_data(x3[frame], y3[frame])
    state0[6] += 0.0001  # Tiny change to vx1

    return trace1, trace2, trace3, point1, point2, point3

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=30, blit=True)
plt.show()
