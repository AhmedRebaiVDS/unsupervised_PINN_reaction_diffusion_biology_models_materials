import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from random import uniform
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)

def omega(a,b,lambd):
  return -b/(8*(4*a+1)*lambd**4)

def xi(x,t,a,b,lambd):
  return lambd*(x-t*lambd**2) + 1/(2*np.log(omega(a,b,lambd)))

def fct_u(x,t,a,b,lambd):
  return (2*lambd**2)/(np.cosh(xi(x,t,a,b,lambd))**2)

def fct_v(x,t,a,b,lambd):
  return 1/(2*np.sqrt(omega(a,b,lambd))*np.cosh(xi(x,t,a,b,lambd)))

a = -1/8
b = -3
lambd = 0.5

N_uv = 1000
N_f = 15000

#Make X_uv_train
#BC x=-10  & tt t > 0
x_left = np.ones((N_uv//4,1), dtype=float)*(-250)
t_left = np.random.uniform(low=0.001, high=10.0, size=(N_uv//4,1))
X_left = np.hstack((x_left, t_left))

#BC x=10 & tt t > 0
x_right = np.ones((N_uv//4,1), dtype=float)*(250)
t_right = np.random.uniform(low=0.001, high=10.0, size=(N_uv//4,1))
X_right = np.hstack((x_right, t_right))

#IC t=0 & tt x,y in [-10,10]
t_zero = np.zeros((N_uv//2,1), dtype=float)
x_zero = np.random.uniform(low=-250.0, high=250.0, size=(N_uv//2,1))
X_zero = np.hstack((x_zero, t_zero))

X_uv_train = np.vstack((X_left, X_right, X_zero))
# shuffling
index=np.arange(0,N_uv)
np.random.shuffle(index)
X_uv_train=X_uv_train[index,:]

#Make u_train
u_left = np.zeros((N_uv//4,1), dtype=float)
u_right = np.zeros((N_uv//4,1), dtype=float)
u_initial = fct_u(x_zero,0,a,b,lambd)
u_train = np.vstack((u_left, u_right, u_initial))

#Make v_train
v_left = np.zeros((N_uv//4,1), dtype=float)
v_right = np.zeros((N_uv//4,1), dtype=float)
v_initial = fct_v(x_zero,0,a,b,lambd)
v_train = np.vstack((v_left, v_right, v_initial))

# ==========================================
u_train=u_train[index,:]
v_train=v_train[index,:]
# ==========================================
# make X_f_train 
X_f_train=np.zeros((N_f,2),dtype=float)
for row in range(N_f):
    x=uniform(-250,250) 
    t=uniform(0,10)  
    X_f_train[row,0]=x
    X_f_train[row,1]=t
   
X_f_train=np.vstack((X_f_train, X_uv_train))

class PhysicsInformedNN():
  def __init__(self,X_uv,u,v,X_f):
    # x & t from boundary conditions
    self.x_uv = torch.tensor(X_uv[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_uv = torch.tensor(X_uv[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    # x & t from collocation points
    self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    # boundary solution
    self.u = torch.tensor(u, dtype=torch.float32)
    self.v = torch.tensor(v, dtype=torch.float32)
    # null vector to test against f:
    self.null =  torch.zeros((self.x_f.shape[0], 1))
    # initialize net:
    self.create_net()

    self.optimizer = torch.optim.Adam(self.net.parameters(),lr=0.001)
    # typical MSE loss (this is a function):
    self.loss = nn.MSELoss()
    # loss :
    self.ls = 0
    # iteration number:
    self.iter = 0

  def create_net(self):
    self.net=nn.Sequential(
        nn.Linear(2,32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        
        nn.Linear(32, 2))
  def net_uv(self,x,t):
    uv=self.net(torch.hstack((x,t)))
    return uv


  def net_fg(self,x,t):
    uv = self.net_uv(x,t)
    u = uv[:,0].reshape(-1,1).to(device)
    v = uv[:,1].reshape(-1,1).to(device)

    #u partial derivatives
    u_t = torch.autograd.grad(u,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    #v partial derivatives
    v_t = torch.autograd.grad(v,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_xxx = torch.autograd.grad(v_xx,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    f = u_t - 6*a*u*u_x -2*b*v*v_x - a*u_xxx
    g = v_t +3*u*v_x + v_xxx

    f = f.to(device)
    g = g.to(device)

    return f,g

  def closure(self):

    # reset gradients to zero 
    self.optimizer.zero_grad()
    
    # u & f predictions
    uv_prediction = self.net_uv(self.x_uv, self.t_uv)
    u_prediction = uv_prediction[:,0].reshape(-1,1)
    v_prediction = uv_prediction[:,1].reshape(-1,1)
    #
    f_prediction_u, f_prediction_v = self.net_fg(self.x_f, self.t_f)

    #
    # losses:
    u_loss = self.loss(u_prediction, self.u)
    v_loss = self.loss(v_prediction, self.v)
    f_loss_u = self.loss(f_prediction_u, self.null)
    f_loss_v = self.loss(f_prediction_v, self.null)

    self.ls = u_loss + v_loss + f_loss_u + f_loss_v
    
    # derivative with respect to net's weights:
    self.ls.backward()

    # increase iteration count:
    self.iter += 1

    # print report:
    if not self.iter % 100:
      print('Epoch: {0:}, Loss: {1:6.8f}'.format(self.iter, self.ls))
      return self.ls    
        
  def train(self):
    for epoch in range(15000):
      self.net.train()
      self.optimizer.step(self.closure)

pinn = PhysicsInformedNN(X_uv_train,u_train,v_train, X_f_train)
pinn.train()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#     
x = torch.linspace(-250, 250, 500)
t = torch.linspace( 0, 10, 500)
# x & t grids:
X,T = torch.meshgrid(x,t)
# x & t columns:
xcol = X.reshape(-1, 1)
tcol = T.reshape(-1, 1)
# one large column:
sol = pinn.net_uv(xcol, tcol)
usol = sol[:,0].reshape(-1,1)
vsol = sol[:,1].reshape(-1,1)
# reshape solution:
U = usol.reshape(x.numel(), t.numel())
V = vsol.reshape(x.numel(), t.numel())
# transform to numpy:
xnp = x.cpu().numpy()
tnp = t.cpu().numpy()
Unp = U.cpu().detach().numpy()
Vnp = V.cpu().detach().numpy()

plt.plot(np.linspace(-250,250,500), fct_u(np.linspace(-250,250,500),100*0.02,a,b,lambd), label='analytic', color="red")
plt.plot(np.linspace(-250,250,500),Unp[:,100], label='pinn', linestyle='dashed')
plt.xlim(-30,30)
plt.legend()

plt.plot(np.linspace(-250,250,500), fct_u(np.linspace(-250,250,500),0.02*300,a,b,lambd), label='analytic', color='red')
plt.plot(np.linspace(-250,250,500),Unp[:,300], label='pinn', linestyle='dashed')
plt.xlim(-50,50)
plt.legend()

plt.plot(np.linspace(-250,250,500), fct_v(np.linspace(-250,250,500),25*0.02,a,b,lambd), label='analytic', color='red')
plt.plot(np.linspace(-250,250,500),Vnp[:,25], label='pinn', linestyle='dashed')
plt.xlim(-50,50)
plt.legend()

x = np.linspace(-250,250,500)
t = np.linspace(0,10,500)

ms_x, ms_t = np.meshgrid(x,t)

U_an = fct_u(ms_x, ms_t, a,b, lambd)
V_an = fct_v(ms_x, ms_t, a,b, lambd)

#RMSE analytic & PINN
np.sqrt(np.mean((Unp-U_an.T)**2))
# 0.00152934398676765

#RMSE analytic & PINN
np.sqrt(np.mean((Vnp-V_an.T)**2))
#0.001979420025691176

a = -1/8
b = -3
lambd = 0.5

delta_x = 1
delta_t = 0.02

x = np.arange(-250,250,delta_x)
t = np.arange(0,10,delta_t)

alpha = delta_t/delta_x
beta = delta_t/(delta_x**3)

n = len(x)
m = len(t)

u_df = np.zeros((n,m))
v_df = np.zeros((n,m))

#For u
#BC
u_df[0,:] = 0
u_df[-1,:] = 0

#IC
u_df[:,0] = fct_u(x,0,a,b,lambd)

#For v
#BC
v_df[0,:] = 0
v_df[-1,:] = 0

#IC
v_df[:,0] = fct_v(x,0,a,b,lambd)

for k in tqdm(range(0,m-1)):
  for i in range(2,n-3):
    u_df[i,k+1] = u_df[i,k] + 6*a*alpha*u_df[i,k]*(u_df[i,k]-u_df[i-1,k]) - 2*b*alpha*v_df[i,k]*(v_df[i,k]-v_df[i-1,k]) - 0.5*a*beta*(u_df[i+2,k]-2*u_df[i+1,k]+2*u_df[i-1,k]-u_df[i-2,k])
    v_df[i,k+1] = v_df[i,k] - 3*alpha*u_df[i,k]*(v_df[i,k]-v_df[i-1,k]) + 0.5*beta*(v_df[i+2,k]-2*v_df[i+1,k]+2*v_df[i-1,k]-v_df[i-2,k]) 

#RMSE PINN & finite difference
np.sqrt(np.mean((Unp-u_df)**2))
# 0.010952436812997277

#RMSE analytic & finite difference
np.sqrt(np.mean((U_an.T-u_df)**2))
# 0.010989140837703614

#RMSE PINN & finite difference
np.sqrt(np.mean((Vnp-v_df)**2))
# 0.006990164623215043

#RMSE analytic & finite difference
np.sqrt(np.mean((V_an-v_df)**2))
# 0.015797690668562524
