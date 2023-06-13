import torch
import numpy as np
import torch.nn as nn
from random import uniform
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)
#
N_uv = 400
N_f = 10000

#Make X_uv_train
#BC x=-1, tt y, t>0
x_left = np.ones((N_uv//8,1), dtype=float)*(-1)
y_left = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//8,1))
t_left = np.random.uniform(low=0.0001, high=10.0, size=(N_uv//8,1))
X_left = np.hstack((x_left, y_left, t_left))

#BC x=1, tt y, t>0
x_right = np.ones((N_uv//8,1), dtype=float)*(1)
y_right = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//8,1))
t_right = np.random.uniform(low=0.0001, high=10.0, size=(N_uv//8,1))
X_right = np.hstack((x_right, y_right, t_right))

#BC y=1, tt x, t>0
x_upper = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//8,1))
y_upper = np.ones((N_uv//8, 1), dtype=float)*(1)
t_upper = np.random.uniform(low=0.0001, high=10.0, size=(N_uv//8,1))
X_upper = np.hstack((x_upper, y_upper, t_upper))

#BC y=-1, tt x, t>0
x_lower = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//8,1))
y_lower = np.ones((N_uv//8,1), dtype=float)*(-1)
t_lower = np.random.uniform(low=0.0001, high=10.0, size=(N_uv//8,1))
X_lower = np.hstack((x_lower, y_lower, t_lower))

#IC t=0, tt x,y
x_zero = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//2,1))
y_zero = np.random.uniform(low=-1.0, high=1.0, size=(N_uv//2,1))
t_zero = np.zeros((N_uv//2,1), dtype=float)
X_zero = np.hstack((x_zero, y_zero, t_zero))

X_uv_train = np.vstack((X_left, X_right, X_upper, X_lower, X_zero))

#Shuffling
index = np.arange(0,N_uv)
np.random.shuffle(index)
X_uv_train = X_uv_train[index,:]

#Make u_train
u_left = np.zeros((N_uv//8,1), dtype=float)
u_right = np.zeros((N_uv//8,1), dtype=float)
u_upper = np.zeros((N_uv//8,1), dtype=float)
u_lower = np.zeros((N_uv//8,1), dtype=float)
u_initial = np.random.uniform(low=0.0, high=1.0, size=(N_uv//2,1))
#u_initial = np.exp(-x_zero**2 - y_zero**2)
u_train = np.vstack((u_left, u_right, u_upper, u_lower, u_initial))

#Make v_train
v_left = np.zeros((N_uv//8,1), dtype=float)
v_right = np.zeros((N_uv//8,1), dtype=float)
v_upper = np.zeros((N_uv//8,1), dtype=float)
v_lower = np.zeros((N_uv//8,1), dtype=float)
v_initial = np.random.uniform(low=0.0, high=1.0, size=(N_uv//2,1))
#v_initial = np.exp(-x_zero**2 - y_zero**2)
v_train = np.vstack((v_left, v_right, v_upper, v_lower, v_initial))

#shuffling
u_train = u_train[index,:]
v_train = v_train[index,:]

#Make X_f_train
X_f_train = np.zeros((N_f,3), dtype=float)
for i in range(N_f):
  x=uniform(-1,1)
  y=uniform(-1,1)
  t=uniform(0,10)
  X_f_train[i,0] = x
  X_f_train[i,1] = y
  X_f_train[i,2] = t

X_f_train = np.vstack((X_f_train, X_uv_train))
#
class PhysicsInformedNN():
  def __init__(self,X_uv,u,v,X_f):
    # x & t from boundary conditions
    self.x_uv = torch.tensor(X_uv[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.y_uv = torch.tensor(X_uv[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_uv = torch.tensor(X_uv[:, 2].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    # x & t from collocation points
    self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.y_f = torch.tensor(X_f[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_f = torch.tensor(X_f[:, 2].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
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
        nn.Linear(3,16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(20, 20), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 2))
    
  def net_uv(self,x,y,t):
    uv=self.net(torch.hstack((x,y,t)))
    return uv


  def net_fg(self,x,y,t):

    a = 2.8e-4
    b = 5e-3
    c = -0.005
    tau=0.1

    uv = self.net_uv(x,y,t)
    u = uv[:,0].reshape(-1,1).to(device)
    v = uv[:,1].reshape(-1,1).to(device)

    
    #u partial derivatives
    u_t = torch.autograd.grad(u,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u,y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y,y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
  

    #v partial derivatives
    v_t = torch.autograd.grad(v,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v,y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y,y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
  

    f = u_t - a*(u_xx + u_yy) - u + u**3 + v - c
    g = v_t - (b*(v_xx + v_yy) + u - v)/tau

    f = f.to(device)
    g = g.to(device)

    return f,g

  def closure(self):

    # reset gradients to zero 
    self.optimizer.zero_grad()
    
    # u & f predictions
    uv_prediction = self.net_uv(self.x_uv, self.y_uv, self.t_uv)
    u_prediction = uv_prediction[:,0].reshape(-1,1)
    v_prediction = uv_prediction[:,1].reshape(-1,1)
    #
    f_prediction_u, f_prediction_v = self.net_fg(self.x_f, self.y_f, self.t_f)

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
    for epoch in range(6000):
      self.net.train()
      self.optimizer.step(self.closure)

pinn = PhysicsInformedNN(X_uv_train,u_train,v_train, X_f_train)
pinn.train()
#     
x = torch.linspace(-1, 1, 200)
y = torch.linspace(-1, 1, 200)
t = torch.linspace( 0, 10, 200)
# x & t grids:
X,Y,T = torch.meshgrid(x,y,t)
# x & t columns:
xcol = X.reshape(-1, 1)
ycol = Y.reshape(-1, 1)
tcol = T.reshape(-1, 1)
# one large column:
sol = pinn.net_uv(xcol, ycol, tcol)
usol = sol[:,0].reshape(-1,1)
vsol = sol[:,1].reshape(-1,1)
# reshape solution:
U = usol.reshape(x.numel(), y.numel(), t.numel())
V = vsol.reshape(x.numel(), y.numel(), t.numel())
# transform to numpy:
xnp = x.cpu().numpy()
ynp = y.cpu().numpy()
tnp = t.cpu().numpy()
Unp = U.cpu().detach().numpy()
Vnp = V.cpu().detach().numpy()

# Plot a heat map for the final result
#plt.imshow(Unp[:,:,-1])  
#plt.imshow(Vnp[:,:,-1])
