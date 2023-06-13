import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

D=1
r=1
delta_x = 0.1
delta_t = 0.001
alpha = D*delta_t/(delta_x**2)
beta = r*delta_t
x = np.arange(-50,50,delta_x)
t = np.arange(0,1,delta_t)
m=len(x)
n=len(t)
u_df = np.zeros((len(t), len(x)))
plt.plot(x, np.heaviside(-x,0))
#Initial condition
u_df[0,:] = np.heaviside(-x,0)

#Bondaries conditions
u_df[:,0] = 1
u_df[:,-1] = 0

#Filling u matrix for finite difference schema
for k in tqdm(range(0,n-1)):
  for i in range(1,m-2):
    u_df[k+1,i] = u_df[k,i] + alpha*(u_df[k,i+1] - 2*u_df[k,i] + u_df[k,i-1]) + beta*u_df[k,i]*(1-u_df[k,i])

plt.plot(x, u_df[100,:])

#Plotting u for finite difference
for i in range(0,1000,200):
  plt.plot(x, u_df[i,:], label='t='+str(i*0.001)+'s')
  plt.xlim(left=-10, right=10)
  plt.legend()
  plt.xlabel('x')
  plt.ylabel('u(x,t)')


# 3D plotting in the phase space
fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(-50,50, delta_x)
t = np.arange(0,1, delta_t)

ms_x, ms_t = np.meshgrid(x,t)


surf = ax.plot_surface(ms_x, ms_t, u_df , cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('t')
#ax.set_zlabel('u(x,t)')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
