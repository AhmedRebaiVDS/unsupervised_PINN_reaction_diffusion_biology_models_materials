import torch
import torch.nn as nn
from random import uniform
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
N_u =2000
N_f =15000

#Make X_u_train
#Boundary condition: x=-10 , tt t>0
x_left = np.ones((N_u//4,1))*(-10)
t_left = np.random.uniform(low=0, high=1, size=(N_u//4,1))
X_left = np.hstack((x_left, t_left))

#Boundary condition: x=10 , tt t>0
x_right = np.ones((N_u//4,1))*(10)
t_right = np.random.uniform(low=0, high=1, size=(N_u//4,1))
X_right = np.hstack((x_right, t_right))

#Initial condition: t=0 , tt x in [-10,10]
x_zero = np.random.uniform(low=-10, high=10, size=(N_u//2,1))
t_zero = np.zeros((N_u//2,1))
X_zero = np.hstack((x_zero, t_zero))

X_u_train=np.vstack((X_right,X_left,X_zero))
# shuffling
index=np.arange(0,N_u)
np.random.shuffle(index)
X_u_train=X_u_train[index,:] 

#Make u_train
# make u_train
u_right=np.zeros((N_u//4,1),dtype=float) # boundary condition x=-10
u_left=np.ones((N_u//4,1),dtype=float)  # boundary condition x=10
u_initial=np.heaviside(-x_zero,0) # initial condition about the function 
u_train=np.vstack((u_right,u_left,u_initial)) # u_train tensor preparation 
u_train=u_train[index,:]

# make X_f_train 
X_f_train=np.zeros((N_f,2),dtype=float)
for row in range(N_f):
  x=uniform(-10,10) 
  t=uniform(0,1)  
  X_f_train[row,0]=x
  X_f_train[row,1]=t
X_f_train=np.vstack((X_f_train, X_u_train)) 

class PhysicsInformedNN():
  def __init__(self,X_u,u,X_f):
    # x & t from boundary conditions
    self.x_u = torch.tensor(X_u[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_u = torch.tensor(X_u[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    # x & t from collocation points
    self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1),dtype=torch.float32,requires_grad=True)
    # boundary solution
    self.u = torch.tensor(u, dtype=torch.float32)
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
        nn.Linear(32, 1))
  def net_u(self,x,t):
    u=self.net(torch.hstack((x,t)))
    return u
  def net_f(self,x,t):
    u=self.net_u(x,t)
    u = u.to(device)

    u_t=torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x=torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx=torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_t = u_t.to(device)
    u_x = u_x.to(device)
    u_xx = u_xx.to(device)
    

    f = u_t - u_xx - u*(1-u)
    f = f.to(device)

    return f

  def closure(self):
    # reset gradients to zero 
    self.optimizer.zero_grad()
    # u & f predictions
    u_prediction = self.net_u(self.x_u, self.t_u)
    #
    f_prediction = self.net_f(self.x_f,self.t_f)
    #
    # losses:
    u_loss_x = self.loss(u_prediction, self.u)
    f_loss = self.loss(f_prediction, self.null)
    self.ls = u_loss_x + f_loss
    # derivative with respect to net's weights:
    self.ls.backward()
    # increase iteration count:
    self.iter += 1
    # print report:
    if not self.iter % 100:
      print('Epoch: {0:}, Loss: {1:6.8f}'.format(self.iter, self.ls))
    return self.ls    
    
  def train(self):
    """
    training loop
    """
    for epoch in range(20000):
      self.net.train()
      self.optimizer.step(self.closure)
     
# pass data sets to the PINN:
pinn = PhysicsInformedNN(X_u_train, u_train, X_f_train)
pinn.train()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#     
x = torch.linspace(-50, 50, 1000)
t = torch.linspace( 0, 1, 1000)
# x & t grids:
X,T = torch.meshgrid(x,t)
# x & t columns:
xcol = X.reshape(-1, 1)
tcol = T.reshape(-1, 1)
# one large column:
usol = pinn.net_u(xcol,tcol)
# reshape solution:
U = usol.reshape(x.numel(),t.numel())
# transform to numpy:
xnp = x.cpu().numpy()
tnp = t.cpu().numpy()
Unp = U.cpu().detach().numpy()

#Plotting u for PINN method
for i in range(0,1000,200):
  plt.plot(np.arange(-50,50,0.1),Unp[:,i], label='t='+str(i*0.001)+'s')
  plt.xlim(left=-10, right=10)
  plt.legend()
  plt.xlabel('x')
  plt.ylabel('u(x,t)')


# A comparison between PINN and the finite difference method
plt.plot(np.arange(-50,50,0.1),Unp[:,500], label='PINN', linestyle='dashed')
plt.plot(np.arange(-50,50,0.1), u_df[500,:], label='Finite Difference', color='red')
plt.xlim(left=-10, right=10)
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,t)')
