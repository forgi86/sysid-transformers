import math
from numpy import zeros, empty, cos, sin, any
from numpy.linalg import solve, LinAlgError
from numpy.random import rand, randn, uniform
from numba import float32, float64, jit, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def drss_matrices_old(
        states, inputs, outputs, strictly_proper=False, positive_real=True, dtype="float32"):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." %
                         states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." %
                         inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." %
                         outputs)

    # Make some poles for A.  Preallocate a complex array.
    poles = zeros(states) + zeros(states) * 0.j
    i = 0

    while i < states:
        if rand() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i:i + 2] = poles[i - 2:i]
                i += 2
        elif rand() < pReal or i == states - 1:
            # No-oscillation pole.
            #
            if positive_real:
                poles[i] = rand()
            else:
                poles[i] = 2. * rand() - 1
            i += 1
        else:
            mag = rand()
            if positive_real:
                phase = math.pi * rand() - math.pi / 2
            else:
                phase = 2. * math.pi * rand()

            poles[i] = complex(mag * cos(phase), mag * sin(phase))
            poles[i + 1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = randn(states, states)
        try:
            A = solve(T, A) @ T  # A = T \ A @ T
            break
        except LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask
        if any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if any(Cmask):  # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else zeros(D.shape)

    return A.astype(dtype), B.astype(dtype), C.astype(dtype), D.astype(dtype)


def drss_matrices(
        states, inputs, outputs, strictly_proper=False, mag_range=(0.2, 0.98), phase_range=(0, math.pi / 2),
        dtype="float64"):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." %
                         states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." %
                         inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." %
                         outputs)

    # Make some poles for A.  Preallocate a complex array.
    poles = zeros(states) + zeros(states) * 0.j
    i = 0

    while i < states:
        if rand() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i:i + 2] = poles[i - 2:i]
                i += 2
        elif rand() < pReal or i == states - 1:
            # No-oscillation pole.
            #
            poles[i] = uniform(mag_range[0], mag_range[1], 1)
            i += 1
        else:
            mag = uniform(mag_range[0], mag_range[1], 1)
            phase = uniform(phase_range[0], phase_range[1], 1)

            poles[i] = complex(mag * cos(phase), mag * sin(phase))
            poles[i + 1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = randn(states, states)
        try:
            A = solve(T, A) @ T  # A = T \ A @ T
            break
        except LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask
        if any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if any(Cmask):  # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else zeros(D.shape)

    return A.astype(dtype), B.astype(dtype), C.astype(dtype), D.astype(dtype)


signatures = [
    float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :]),
    float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :])
]
#@jit(float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :]), nopython=True,
#     cache=True)
@jit(signatures, nopython=True, cache=True)
def dlsim(A, B, C, D, u):
    # A, B, C, D = sys_matrices
    seq_len = u.shape[0]
    nx, nu = B.shape
    ny, _ = C.shape
    y = empty(shape=(seq_len, ny), dtype=u.dtype)
    x_step = zeros((nx,), dtype=u.dtype)
    for idx in range(seq_len):
        u_step = u[idx]
        y[idx] = C.dot(x_step) + D.dot(u_step)
        x_step = A.dot(x_step) + B.dot(u_step)
    return y
