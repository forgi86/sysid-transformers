import math
from numpy import any, cos, sin, zeros
from numpy.random import rand, randn
from numpy.linalg import solve
from numpy.linalg.linalg import LinAlgError
from lti import drss_matrices_old, drss_matrices



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TKAgg")
    import numpy as np
    import control
    import matplotlib.pyplot as plt
    #A, B, C, D = random_ss(5, 1, 1, positive_real=True)
    A, B, C, D = drss_matrices(100, 1, 1)
    E, V = np.linalg.eig(A)

    theta = np.linspace(0, 2*math.pi, 1000)
    plt.figure(figsize=(5, 5))
    plt.plot(E.real, E.imag, "*")
    plt.plot(cos(theta), sin(theta))
    plt.show()

    sys = control.ss(A, B, C, D, dt=1.0)

    plt.figure()
    y_imp = control.impulse_response(sys)
    plt.plot(y_imp.t, y_imp.y.ravel(), "--*")

    plt.figure()
    control.bode_plot(sys)
