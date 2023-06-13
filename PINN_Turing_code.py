import torch
import numpy as np
import torch.nn as nn
from random import uniform, random
import random
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(device)

rb = 0.0347
rc = 2
db = 10**(0)
dc = 10**(5)
bi = 10**(7)
k=0.1
fb = k*rc
alpha = 0.3129
sb = 10**(5)
theta = 0.3
fe = alpha*theta*bi/((sb+theta*bi)*(1-theta))-rb/k

N_uv = 14000
N_f =  20000

#Make X_uv_train
#BC x=-10  & tt t > 0
x_left = np.zeros((N_uv//4,1), dtype=float)
t_left = np.random.uniform(low=0.0, high=1500.0, size=(N_uv//4,1))
X_left = np.hstack((x_left, t_left))

#BC x=10 & tt t > 0
x_right = np.ones((N_uv//4,1), dtype=float)*(3000)*10**(-6)
t_right = np.random.uniform(low=0.0, high=1500.0, size=(N_uv//4,1))
X_right = np.hstack((x_right, t_right))

#IC t=0 & tt x,y in [-10,10]
t_zero = np.zeros((N_uv//2,1), dtype=float)
x_zero = np.random.uniform(low=0.0, high=3000*10**(-6), size=(N_uv//2,1))
X_zero = np.hstack((x_zero, t_zero))

X_uv_train = np.vstack((X_left, X_right, X_zero))
# shuffling
index=np.arange(0,N_uv)
np.random.shuffle(index)
X_uv_train=X_uv_train[index,:]

#Make u_train
u_left = np.zeros((N_uv//4,1), dtype=float)
u_right = np.zeros((N_uv//4,1), dtype=float)
#u_initial = list_indicator(np.sort(x_zero, axis=0),1400*10**(-6),1600*10**(-6)).reshape(N_uv//2,1)*10**(5)
#u_initial = np.zeros((N_uv//2,1), dtype=float)
#u_initial[N_uv//4 -50: N_uv//4 +50] = 10**(5)
u_initial = np.exp(-((x_zero*10**(6) - N_uv//4)**2)/10000)*10**(5)
u_train = np.vstack((u_left, u_right, u_initial))


#Make v_train
v_left = np.zeros((N_uv//4,1), dtype=float)
v_right = np.zeros((N_uv//4,1), dtype=float)
v_initial = np.zeros((N_uv//2,1), dtype=float)
v_train = np.vstack((v_left, v_right, v_initial))

# ==========================================
u_train=u_train[index,:]
v_train=v_train[index,:]
# ==========================================
# make X_f_train 
X_f_train=np.zeros((N_f,2),dtype=float)
for row in range(N_f):
    #x=random.randint(0,3000)*10**(-6) 
    x = uniform(0,3000*10**(-6))
    #t=random.randint(0,5000)  
    t = uniform(0,1500)
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
        nn.Linear(2,16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(16, 16), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 32), nn.Sigmoid(),
        nn.Linear(32, 2))
        
  def net_uv(self,x,t):
    uv=self.net(torch.hstack((x,t)))
    return uv


  def net_fg(self,x,t):

    rb = 0.0347
    rc = 2
    db = 10**(0)
    dc = 10**(5)
    bi = 10**(7)
    k=0.1
    fb = k*rc
    alpha = 0.3129
    sb = 10**(5)
    theta = 0.3
    fe = alpha*theta*bi/((sb+theta*bi)*(1-theta))-rb/k

    uv = self.net_uv(x,t)
    u = uv[:,0].reshape(-1,1).to(device)
    v = uv[:,1].reshape(-1,1).to(device)

    #u partial derivatives
    u_t = torch.autograd.grad(u,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    

    #v partial derivatives
    v_t = torch.autograd.grad(v,t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x,x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    

    f = u_t - db*u_xx - rb*(1 - u/bi)*u + (alpha*u*v)/(sb + u) - fe*(1 - u/bi)*v
    g = v_t - dc*v_xx - fb*u + rc*v

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
      print('Epoch: {0:}, Loss: {1:6.12f}'.format(self.iter, self.ls))
      return self.ls    
        
  def train(self):
    for epoch in range(15000):
      self.net.train()
      self.optimizer.step(self.closure)
