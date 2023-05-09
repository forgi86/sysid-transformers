import math
from numpy import cos, sin
from numpy.random import rand
from numpy.linalg import solve
from lti import drss_matrices, dlsim, drss_matrices_old


if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use("TKAgg")
    import control
    import matplotlib.pyplot as plt
    import time

    seq_len = 1000
    nu = 1
    dtype = "float64"
    time_start = time.time()
    for _ in range(1000):
        A, B, C, D = drss_matrices(20, 1, 1, dtype=dtype)#, positive_real=True)

        u = np.random.randn(seq_len, nu).astype(dtype)
        y = dlsim(A, B, C, D, u)

    G1 = control.ss(A, B, C, D, dt=1.0)
    y_ct = control.forced_response(G1, T=None, U=u.transpose(), X0=0.0)
    y_ct = y_ct.y.astype("float32").transpose()  # T, C
    assert(np.allclose(y, y_ct, rtol=1e-5, atol=1e-7))

    time_op = time.time() - time_start
    print(f"{time_op=:.2f}")
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

    plt.figure()
    plt.plot(y)
