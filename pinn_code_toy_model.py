import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden_layer1 = nn.Linear(2,5)
    self.hidden_layer2 = nn.Linear(5,5)
    self.hidden_layer3 = nn.Linear(5,5)
    self.hidden_layer4 = nn.Linear(5,5)
    self.hidden_layer5 = nn.Linear(5,5)
    self.output_layer = nn.Linear(5,1)

  def forward(self, x, t):
    inputs = torch.cat([x,t], axis=1)
    layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
    layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
    layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
    layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
    layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
    output = self.output_layer(layer5_out)

    return output

def f(x, t, net):
  u = net(x, t)
  u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
  u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]

  pde = u_x - 2*u_t - u 
  return pde

def get_boundary_data():
  x_bc = np.random.uniform(low=0.0, high=2.0, size=(500,1))
  t_bc = np.zeros((500,1))

  u_bc = 6*np.exp(-3*x_bc)
  return x_bc, t_bc, u_bc

def get_collocation_data():
  x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
  t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
  all_zeros = np.zeros((500,1))
  return x_collocation, t_collocation, all_zeros

def train_model(net, cost_function, optimizer, num_iterations):
  previous_validation_loss = 99999999.0
  
  for epoch in range(num_iterations):
    optimizer.zero_grad()

    # Loss based on boundaries conditions
    x_bc, t_bc, u_bc = get_boundary_data()
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)

    net_bc_out = net(pt_x_bc, pt_t_bc)
    mse_u = cost_function(net_bc_out, pt_u_bc)

    # Loss based on PDE
    x_collocation, t_collocation, all_zeros = get_collocation_data()
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, pt_t_collocation, net)
    mse_pde = cost_function(f_out, pt_all_zeros)

    # Total loss
    total_loss = mse_u + mse_pde
    total_loss.backward()
    optimizer.step()

    # Validation
    if epoch % 1000 == 0:
      x_validation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
      t_validation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
      u_validation = 6*np.exp(-3*x_validation)

      pt_x_validation = Variable(torch.from_numpy(x_validation).float(), requires_grad=False).to(device)
      pt_t_validation = Variable(torch.from_numpy(t_validation).float(), requires_grad=False).to(device)
      pt_u_validation = Variable(torch.from_numpy(u_validation).float(), requires_grad=False).to(device)

      net_validation_out = net(pt_x_validation, pt_t_validation)
      mse_validation_loss = cost_function(net_validation_out, pt_u_validation)
