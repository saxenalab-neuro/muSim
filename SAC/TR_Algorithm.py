import numpy as np 
import matplotlib.pyplot as plt 
import time

## ------- Functions related to finite differencing -------------

#Parameters for the optimization algorithm
delta_step= 1e-4

def gradient_fd(f, x):
  num_variables = len(x)
  out = [ ]

  f_0 = f(x)
  delta_x = delta_step

  for i in range(num_variables):
    x_copy = np.copy(x)
    x_copy[i] = x_copy[i] + delta_x
    f_delta = f(x_copy)
    out.append( (f_delta - f_0) / delta_x )

  return np.array(out)

def hessian_fd(f, x):
  num_variables = len(x)
  out = [ ]

  f_0 = gradient_fd(f, x)
  delta_x = delta_step

  for i in range(num_variables):
    x_copy = np.copy(x)
    x_copy[i] = x_copy[i] + delta_x
    f_delta = gradient_fd(f, x_copy)
    out.append( (f_delta - f_0) / delta_x )

  return np.array(out)

def fs(x):
  return f_obj(x)

def Dfs(x):
  return gradient_fd(fs, x)

def Hfs(x):
  return hessian_fd(fs, x)


#maps the intial state to the final state that minimizes the obj func
def TR_Algorithm(obj_func, initial_state, env):

  #Parameters for the optimization algorithm
  tol= 5e-04
  r0= 0.001
  max_iters = 10000


  global f_obj
  f_obj = obj_func
  
  x0 = initial_state

  iter_t = 0

  #whether the optimization was successful or not
  success = False
  cum_loss = 0
  while(1):

      pks = -1*r0*Dfs(x0).T/np.linalg.norm(Dfs(x0).T)

      if Dfs(x0)@Hfs(x0)@Dfs(x0).T <= 0:
          tk = 1
      else:
          thresh = np.linalg.norm(Dfs(x0))**3 / (  r0*Dfs(x0)@Hfs(x0)@Dfs(x0).T )
          tk = min(1, thresh)

      pkc = tk*pks

      x1 = x0 + pkc

      #Advance the simulator by one step as well
      env.set_state_musculo(x1)
      env.render()

      x0 = x1
      r0 = r0

      # if np.linalg.norm(Dfs(x0)) < tol and np.linalg.norm(fs(x0)) < tol:
      if np.linalg.norm(fs(x0)) < tol:

        #Return the final qpos and qvel

        state_opt = env.data.qpos.flat.copy()
        state_opt_musculo = env.get_musculo_state()
        final_loss = np.linalg.norm(fs(x0))
        success = True

        return state_opt, state_opt_musculo, final_loss, cum_loss, success

      iter_t += 1
      #add to the cumulative loss
      cum_loss += np.linalg.norm(fs(x0))

      if iter_t >= max_iters:
        print('Max iters achieved for the IK algorithm')
        print('Try the following: 1. Decrease tol (vars) in IK algorithm')
        print('2. Change other parameters including r0, delta step')
        print('3. Use MuJoCo to provide a better initial pose')
        
        state_opt = env.data.qpos.flat.copy()
        state_opt_musculo = env.get_musculo_state()
        final_loss = np.linalg.norm(fs(x0))
        success = False

        return state_opt, state_opt_musculo, final_loss, cum_loss, success