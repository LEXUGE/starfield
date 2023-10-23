import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt

FIG_PATH = "outputs/realspace_plot_%i.svg"


# Our time-independent Hamiltonian H(q,p)
def H(w):
    q = (w[0], w[1])
    p = (w[2], w[3])
    return 0.5 * (p[0] * p[0] + p[1] * p[1]) + EffPotential(q)


# Axissymmetrical Effective Potential
def EffPotential(q):
    return 0.5 * (
        q[0] * q[0] + q[1] * q[1] + 2 * q[0] * q[0] * q[1] - 2 * q[1] * q[1] * q[1] / 3
    )


# p dot  = - partial H / partial q = - partial EffPotential / partial q
def Force(q):
    return (-q[0] * (1 + 2 * q[1]), -q[1] - q[0] * q[0] + q[1] * q[1])


def HamiltonEqns(t, w):
    q = (w[0], w[1])
    f = Force(q)
    # q dot = partial H / partial p = p in this case
    return (w[2], w[3], f[0], f[1])


# Register an event when w[0] = 0, which means z = 0
# t in signature is necessary as required by scipy
def xcross(t, w):
    return w[0]


def add_orbit(H0, q0):
    # v0 = sqrt(p_z^2 + p_r^2)
    v0 = np.sqrt(2 * (H0 - EffPotential(q0)))
    theta = -0.3
    p0 = (np.cos(theta) * v0, np.sin(theta) * v0)
    # initial phase space position
    w = (q0[0], q0[1], p0[0], p0[1])
    print("H at t0:", H(w))

    # timespan, t_0 and t_f
    # set a smaller t_f to avoid a overcrowded plot in realspace.
    ts = (0, 300)
    # absolute tolerances in numerical integration
    acc = (1e-10, 1e-10, 1e-10, 1e-10)
    # scipy doc:
    # Direction of a zero crossing. If direction is positive, event will only trigger when going from negative to positive
    xcross.direction = 1
    soln = solve(
        HamiltonEqns,
        ts,
        w,
        method="RK45",
        events=xcross,
        rtol=1e-5,
        atol=acc,
        dense_output=True,
        t_eval=None,
    )
    # Number of time steps taken?
    n = soln.t.size - 1
    # The final state at t_f
    w = (soln.y[0, n], soln.y[1, n], soln.y[2, n], soln.y[3, n])
    print("H at t_f:", H(w))

    print(n, " points")

    r_trajectories = []
    z_trajectories = []
    for i in range(0, n, 1):
        r_trajectories.append(soln.y[0, i])
        z_trajectories.append(soln.y[1, i])

    plt.plot(r_trajectories, z_trajectories, linewidth=0.5)


def main():
    # Energy, or the constant of Hamiltonian
    H0 = 0.08333
    # Number of orbits to plot
    norb = 10
    for i in range(0, norb, 1):
        plt.clf()
        plt.title(
            "Real Space Projection onto r-z plane with initial r = %1.2f"
            % (-0.3 + 0.5 * i / (norb - 1))
        )
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        plt.xlabel("r")
        plt.ylabel("z")
        # q[0] is z, q[1] is r
        # change the value of initial r from -0.3 to 0.2, with adaptive steps based on `norb`.
        init_q = (0, -0.3 + 0.5 * i / (norb - 1))
        add_orbit(H0, init_q)

        plt.savefig(FIG_PATH % i)


main()
