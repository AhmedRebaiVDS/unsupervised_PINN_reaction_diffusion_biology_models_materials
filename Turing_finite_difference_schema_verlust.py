import numpy as np
import matplotlib.pyplot as plt

def reac_diff_solver_1D():
    L = 3000
    N = 3000
    T = 200000
    dt = 1

    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    t = np.arange(0, T+dt, dt)
    t_num = len(t)

    d_b = 10**0
    d_c = 10**5

    # r_c = 0.0002 # To get a Turing process
    r_c = 2  # This value kill the Turing process
    b_i = 10**17  # To get a Turing process
    r_b = 0.0347
    k = 0.1  # (c*=k*b*)
    f_b = k*r_c

    s_b = 10**15
    theta = 0.3  # (b*=theta*b_i)
    alpha = 0.3129

    f_e = alpha*theta*b_i/((s_b+theta*b_i)*(1-theta))-r_b/k

    Kb = dt/(dx**2)*d_b
    Kc = dt/(dx**2)*d_c

    b0 = 1*10**15*(1495 < x)*(x < 1505)

    c0 = np.zeros_like(b0)

    Ab = np.diag((1+2*Kb)*np.ones(N)) - np.diag(Kb*np.ones(N-1), 1) - np.diag(Kb*np.ones(N-1), -1)
    Ab[0, 0] = 1 + Kb
    Ab[-1, -1] = 1 + Kb

    Ac = np.diag((1+2*Kc+dt*r_c)*np.ones(N)) - np.diag(Kc*np.ones(N-1), 1) - np.diag(Kc*np.ones(N-1), -1)
    Ac[0, 0] = 1 + Kc + dt*r_c
    Ac[-1, -1] = 1 + Kc + dt*r_c

    B = np.zeros((t_num, N))
    B[0, :] = b0

    C = np.zeros((t_num, N))
    C[0, :] = c0

    source_b = np.zeros(N)
    source_c = np.zeros(N)
    MD_b = np.zeros(N)
    MD_c = np.zeros(N)
    
    plt.figure()
    plt.plot(10**(-6)*x, B[0, :], "blue", 10**(-6)*x, C[0, :], "red")
    for j in range(1, t_num):
        
        MD_b = B[j-1, :] + dt*f_e*C[j-1, :]*(1 - B[j-1, :]/b_i) + dt*r_b*(1 - B[j-1, :]/b_i)*B[j-1, :] - dt*alpha*B[j-1, :]*C[j-1, :]/(s_b + B[j-1, :])
        Ab_extended = Ab
        B[j, :] = np.linalg.solve(Ab_extended, MD_b)
        MD_c = C[j-1, :] + dt*f_b*B[j, :]
        C[j, :] = np.linalg.solve(Ac, MD_c)

        plt.plot(10**(-6)*x, B[j, :], "blue", 10**(-6)*x, C[j, :], "red")
        plt.xlim([0, 3*10**(-3)])
        plt.ylim([0, 10**(17)])
        plt.legend(["bacteria at time t="+str(t[j])])
        plt.pause(0.00001)

    plt.figure()
    plt.plot(10**(-6)*x, B[-1, :], "blue", 10**(-6)*x, 100*C[-1, :], "red", 10**(-6)*x, B[0, :], "yellow")
    plt.xlabel("spatial variable $x$ (meters)")
    plt.ylabel("temporal variables")
    plt.legend(["bacteria", "chemoattractant", "initial bacteria"])

reac_diff_solver_1D()
