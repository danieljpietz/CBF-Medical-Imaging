## Made by Daniel Pietz for Bionaut Labs

import numpy as np
from improc import loadmap, plot
from cbf import cbf_eval, interp_eval2d
import time as tm


class Model:
    def __init__(self):
        self.drag_coeff = 0.5
        self.drag_coeff2 = 0.5
        self.mass = 0.01
        self.size = 3
        self._barrier, self._g_barrier, self._h_barrier = loadmap(
            image, thresh=thresh, smooth=smooth
        )

    def f(self, s):  # Environmental forces, in this case just Coloumbic friction
        x, y, theta, dot_x, dot_y, dot_theta = tuple(s)
        f_drag = -self.drag_coeff * np.array([dot_x, dot_y, 0])
        rot_drag = -self.drag_coeff2 * np.array([0, 0, dot_theta])
        return f_drag + rot_drag

    def g_f(self, s):  # Gradient of Environmental forces
        f_drag = -self.drag_coeff * np.array([1, 1, 0])
        rot_drag = -self.drag_coeff2 * np.array([0, 0, 1])
        return f_drag + rot_drag

    def h_f(self, s):  # Hessian matrix of Environmental forces
        return np.zeros((3, 3))

    def g(self, s):  # input mapping matrix
        return np.eye(3) / self.mass

    def barrier(self, s):  # CBF
        self.b = interp_eval2d(self._barrier, s) - 80
        return self.b

    def g_barrier(self, s):  # d/dx CBF
        return np.pad(interp_eval2d(self._g_barrier, s), (0, 1), "constant")

    def dot_barrier(self, s):  # d/dt CBF
        return np.dot(self.g_barrier(s), np.array([s[3], s[4], s[5]]))

    def h_barrier(self, s):  # d^2/dx^2 CBF
        return np.pad(interp_eval2d(self._h_barrier, s), (0, 1), "constant")

    def cbf_filter(self, s, u_d):  # Evaluate the CBF with a given desired control input
        return cbf_eval(
            (3, self.f(s), self.g_f(s), self.g(s)),
            (
                self.barrier(s),
                self.g_barrier(s),
                self.h_barrier(s),
                self.dot_barrier(s),
            ),
            u_d,
            lam=(L1, L2),
        )


# Load the image
image = "images/Artery.png"

# Hyper parameters for the CBF. Generally a lower value means a stricter CBF

L1 = 10
L2 = 10

# Image thresholding and smoothing values
thresh = 90
smooth = 1

# Starting position

start_x = 1150
start_y = 723
start_dx = -100
start_dy = 10

# Time step

h = 0.05

# Run the simulation

s0 = np.array([start_x, start_y, 0, start_dx, start_dy, 0], dtype=float)

microbot = Model()
s = s0
plot(s)
t_start = tm.time()
for i in range(1000):
    u_safe = microbot.cbf_filter(s, np.array([-0, 0, 0]))
    accel = microbot.f(s) + microbot.g(s) @ u_safe
    s[:3] += h * s[3:]
    s[3:] += h * accel
    plot(s)

plot(s, True)
