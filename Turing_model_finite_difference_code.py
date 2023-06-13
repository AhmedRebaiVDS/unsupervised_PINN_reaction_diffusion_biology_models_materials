import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reac_diff_solver_2D():
    L = 3000
    N = 100
    T = 20000

    x = np.linspace(0, L, N)
    y = x
    dx = np.max(np.abs(np.diff(x)))
    dt = 1
    R = 0.5
    tempIt = int(T/dt)

    Xi, Yi = np.meshgrid(x, y)

    d_b = 10**0
    d_c = 10**3
    r_c = 0.02
    b_i = 10**17
    r_b = 0.0347
    k = 0.1
    f_b = k*r_c
    s_b = 10**15
    theta = 0.3
    alpha = 0.3129

    f_e = alpha*theta*b_i/((s_b+theta*b_i)*(1-theta))-r_b/k

    Rb = dt/(2*dx**2)*d_b
    Rc = dt/(2*dx**2)*d_c

    b0 = 10**15*np.double(np.abs(Xi-L/2)<50)
    c0 = 0*b0

    X = np.reshape(Xi, (N)**2, 1)
    Y = np.reshape(Yi, (N)**2, 1)

    B_aux = np.array([b0[:,0], b0, b0[:, -1]])
    B_aux = np.vstack([B_aux[0,:], B_aux, B_aux[-1,:]])
    B_vector = np.reshape(B_aux, (N+2)**2, 1)
    C_aux = np.array([c0[:,0], c0, c0[:, -1]])
    C_aux = np.vstack([C_aux[0,:], C_aux, C_aux[-1,:]])
    C_vector = np.reshape(C_aux, (N+2)**2, 1)

    t = 0

    Mb = np.diag((1+2*Rb)*np.ones((N+2)**2,1)) - np.diag(Rb*np.ones((N+2)**2-1,1), -1) - np.diag(Rb*np.ones((N+2)**2-1,1), 1)
    Mb = np.asarray(Mb).tolist()
    Mc = np.diag((1+2*Rc+dt*r_c)*np.ones((N+2)**2,1)) - np.diag(Rc*np.ones((N+2)**2-1,1), -1) - np.diag(Rc*np.ones((N+2)**2-1,1), 1)
    Mc = np.asarray(Mc).tolist()
    Ab = np.diag((1-2*Rb)*np.ones((N+2)**2,1)) + np.diag(Rb*np.ones((N+2)**2-(N+2),1), (N+2)) + np.diag(Rb*np.ones((N+2)**2-(N+2),1), -(N+2))
    Ab = np.asarray(Ab).tolist()
    Ac = np.diag((1-2*Rc)*np.ones((N+2)**2,1)) + np.diag(Rc*np.ones((N+2)**2-(N+2),1), (N+2)) + np.diag(Rc*np.ones((N+2)**2-(N+2),1), -(N+2))
    Ac = np.asarray(Ac).tolist()

    Mbinv = np.linalg.inv(Mb)
    Gb = Mbinv*Ab
    Mcinv = np.linalg.inv(Mc)
    Gc = Mcinv*Ac

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121, projection='3d')
    bsurf = ax1.plot_surface(Xi, Yi, b0)
    ax2 = fig.add_subplot(122, projection='3d')
    csurf = ax2.plot_surface(Xi, Yi, c0)
    ptit = plt.title(f't = {t}')
    plt.xlim([0, L])
    plt.ylim([0, L])

    for it in range(1, tempIt+1):
        t = t+dt
        B_vector = Gb*B_vector
        C_vector = Gc*C_vector

        B_aux = np.reshape(B_vector, (N+2, N+2))
        C_aux = np.reshape(C_vector, (N+2, N+2))

        B_aux[1:N+1, 1:N+1] = B_aux[1:N+1, 1:N+1] + dt*(f_b*C_aux[1:N+1, 1:N+1] - theta*B_aux[1:N+1, 1:N+1] + b_i*(1-theta))
        C_aux[1:N+1, 1:N+1] = C_aux[1:N+1, 1:N+1] + dt*(r_c*C_aux[1:N+1, 1:N+1] + f_e*B_aux[1:N+1, 1:N+1])

        B_vector = np.reshape(B_aux, (N+2)**2, 1)
        C_vector = np.reshape(C_aux, (N+2)**2, 1)

    if it % 50 == 0:
        bsurf.set_zdata(np.reshape(B_vector, (N+2, N+2))[1:N+1, 1:N+1])
        csurf.set_zdata(np.reshape(C_vector, (N+2, N+2))[1:N+1, 1:N+1])
        ptit.set_text(f't = {t}')
        plt.draw()
        plt.pause(0.001)
        plt.show()    
