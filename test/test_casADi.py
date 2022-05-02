# Car race along a track
# ----------------------
# An optimal control problem (OCP),
# solved with direct multiple-shooting.
#
# For more information see: http://labs.casadi.org/OCP
from casadi import *
import math as m

N = 100 # number of control intervals

opti = Opti() # Optimization problem

# ---- decision variables ---------
# Putting in obstacles and target
obs1 = [9.0, 25.0, 4.5]
obs2 = [20.0, 35.0, 3.0]
obs3 = [6.5, 50.0, 3.0]
obs4 = [-4.0, 33.0, 2.0]
obs5 = [-5.0, 44.0, 2.0]
target = [6.0, 40.0, 1.96]


X = opti.variable(5,N+1) # state trajectory
pos_x   = X[0,:]
pos_y = X[1,:]
theta = X[2,:]
phi = X[3,:]
speed = X[4,:]

U = opti.variable(2,N)   # control trajectory (throttle)
angular_rate = U[0,:]
accel = U[1,:]

#T = opti.variable()      # final time
T = 10

# ---- objective          ---------
a = sqrt(2)
print(a)
dx = pos_x[N] - target[0]
dy = pos_y[N] - target[1]
relative_dist = sqrt(dx*dx + dy*dy)
opti.minimize(relative_dist - target[2])
#opti.minimize(T) # race in minimal time

# ---- dynamic constraints --------
f = lambda x,u: vertcat(x[4]*np.cos(x[2]), x[4]*np.sin(x[2]), x[4]*np.tan(x[3]), u[0], u[1])

dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
limit = lambda pos: 1-sin(2*pi*pos)/2
#opti.subject_to(speed<=limit(pos))   # track speed limit
#opti.subject_to(opti.bounded(0,U,1)) # control is limited
opti.subject_to(opti.bounded(-25,U,25)) # control is limited

# ---- boundary conditions --------
opti.subject_to(pos_x[0]==2)   # start at position x = 0 ...
opti.subject_to(pos_y[0]==0)   # start at position y = 0 ...
opti.subject_to(theta[0] == pi/2)   # start at theta = 0 ...
opti.subject_to(phi[0]==0)  # start at phi = 0 ...
opti.subject_to(speed[0]==3) # ... from stand-still 
#opti.subject_to(pos_x[-1]==1)  # finish line at position 1

# ---- misc. constraints  ----------
#opti.subject_to(T>=0) # Time must be positive




# ---- initial values for solver ---
opti.set_initial(speed, 1)
#opti.set_initial(T, 10)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve

# ---- post-processing        ------
#from pylab import plot, step, figure, legend, show, spy

#plot(sol.value(speed),label="speed")
#plot(sol.value(pos),label="pos")
#plot(limit(sol.value(pos)),'r--',label="speed limit")
#step(range(N),sol.value(U),'k',label="throttle")
#legend(loc="upper left")

#figure()
#spy(sol.value(jacobian(opti.g,opti.x)))
#figure()
#spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))

#show()
