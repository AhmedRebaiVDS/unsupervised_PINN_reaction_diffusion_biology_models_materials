v = 1
r = 1/4
t_f = 0.1
delta_x = 0.01
delta_t = (r*delta_x**2)/v
N =  int(1/delta_x)
J = int(t_f/delta_t)

print('delta_x =', delta_x,'delta_t =',delta_t,'N =',N,'J =',J)

x = np.arange(0, 1, delta_x)
t = np.arange(0, t_f, delta_t)

theta_df = np.zeros((N,J))

#Initial condition
theta_df[:,0] = A_0(x, v)

for j in tqdm(range(0,J-1)):
  theta_df[0,j+1] = (1 - 2*r)*theta_df[0,j] + 2*r*theta_df[1,j]
  for i in range(1,N-1):
    theta_df[i,j+1] = r*theta_df[i-1,j] + (1 - 2*r)*theta_df[i,j] + r*theta_df[i+1,j]

  theta_df[N-1,j+1] = 2*r*theta_df[N-2,j] + (1 - 2*r)*theta_df[N-1,j]

u_df = np.zeros((N-2, J))

for j in tqdm(range(0,J)):
  for i in range(1,N-2):
    u_df[i,j] = -(v/delta_x)*(theta_df[i+1,j] - theta_df[i-1,j])/theta_df[i,j]

print(u_df.shape)

for i in range(0,J,500):
  plt.plot(u_df[:,i], label = 'i =' +str(i*delta_t))
  plt.legend()

for i in range(0,J,500):
  plt.plot(fct_u(x, i*delta_t,1), label = 'i =' +str(i*delta_t))
  plt.legend()
