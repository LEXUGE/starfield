import numpy as np
from scipy.integrate import solve_ivp as solve
import matplotlib.pyplot as plt

FIG_PATH = "outputs/log_realspace/plot_%i.svg"
V0 = 1
R_c = 0.14
Q = 0.6

# z is a cyclic coordinate and indeed ignorable here. We set p_z = 0 so z stays at some constant value.


# Our time-independent Hamiltonian H(q,p)
# q[0] = r, q[1] = \phi
def H(w):
    q = (w[0], w[1])
    p = (w[2], w[3])
    return 0.5 * (p[0] ** 2 + (p[1] / q[0]) ** 2) + LogPotential(q)


def LogPotential(q):
    R_sqr = q[0] ** 2
    Q_2 = Q ** (-2)
    return (
        0.5
        * (V0**2)
        * np.log(
            R_c**2
            + 0.5 * R_sqr * (Q_2 + 1)
            - 0.5 * R_sqr * (Q_2 - 1) * np.cos(2 * q[1])
        )
    )


# p dot  = - partial H / partial q = - partial EffPotential / partial q
def pDot(q, p):
    R_sqr = q[0] ** 2
    Q_2 = Q ** (-2)

    p_phi_Dot = (
        -0.5
        * (
            (V0**2)
            * (
                R_c**2
                + 0.5 * R_sqr * (Q_2 + 1)
                - 0.5 * R_sqr * (Q_2 - 1) * np.cos(2 * q[1])
            )
            ** (-1)
        )
        * R_sqr
        * (Q_2 - 1)
        * np.sin(2 * q[1])
    )
    p_r_Dot = -0.5 * (V0**2) * (
        (
            R_c**2
            + 0.5 * R_sqr * (Q_2 + 1)
            - 0.5 * R_sqr * (Q_2 - 1) * np.cos(2 * q[1])
        )
        ** (-1)
    ) * (q[0] * (Q_2 + 1) - q[0] * (Q_2 - 1) * np.cos(2 * q[1])) + (p[1] ** 2) * (
        q[0] ** (-3)
    )

    return (p_r_Dot, p_phi_Dot)


def HamiltonEqns(t, w):
    q = (w[0], w[1])
    p = (w[2], w[3])
    f = pDot(q, p)
    return (w[2], w[3] / (w[0] ** 2), f[0], f[1])


def cross(t, w):
    return w[1]


def add_orbit(label, H0, q0):
    v0 = np.sqrt(2 * (H0 - LogPotential(q0)))
    theta = -0.3
    p0 = (np.cos(theta) * v0, np.sin(theta) * v0 * q0[0])
    # initial phase space position
    w = (q0[0], q0[1], p0[0], p0[1])
    print("H at t0:", H(w))

    # timespan, t_0 and t_f
    ts = (0, 300)
    # absolute tolerances in numerical integration
    acc = (1e-10, 1e-10, 1e-10, 1e-10)
    # scipy doc:
    # Direction of a zero crossing. If direction is positive, event will only trigger when going from negative to positive
    cross.direction = 1
    soln = solve(
        HamiltonEqns,
        ts,
        w,
        method="RK45",
        events=cross,
        rtol=1e-10,
        atol=acc,
        dense_output=True,
        t_eval=None,
    )
    # Number of time steps taken?
    n = soln.t.size - 1
    # The final state at t_f
    w = (soln.y[0, n], soln.y[1, n], soln.y[2, n], soln.y[3, n])
    print("H at t_f:", H(w))

    r_trajectories = []
    phi_trajectories = []
    for i in range(0, n, 1):
        r_trajectories.append(soln.y[0, i])
        phi_trajectories.append(soln.y[1, i])

    plt.plot(r_trajectories, phi_trajectories, linewidth=0.5)


def main():
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(-0.5, 0.5)

    # Energy, or the constant of Hamiltonian
    H0 = 0.08333
    # Number of orbits to plot
    norb = 10
    for i in range(0, 2, 1):
        plt.clf()
        plt.title(
            "Real Space Projection onto r-phi plane with initial r = %1.2f"
            % (-0.3 + 0.5 * i / (norb - 1))
        )
        plt.xlabel("r")
        plt.ylabel("phi")

        init_q = (0.2, -0.3 + 0.5 * i / (norb - 1))
        add_orbit(-0.3 + 0.5 * i / (norb - 1), H0, init_q)

        plt.savefig(FIG_PATH % (i + 1))


main()
