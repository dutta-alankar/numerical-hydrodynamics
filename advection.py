# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 11:27:07 2020

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt

class advection:
    
    def __init__(self, x, ic, t_start, t_stop, t_step, cfl, scheme, bc, a):
        self.u = np.concatenate((np.array([0,0]),ic,np.array([0,0]))) #Put ghosts (till 2nd order)
        self.x = np.array(x)
        self.t_start = t_start
        self.t_stop = t_stop
        self.t_step = t_step
        self.scheme = scheme
        self.bc = bc
        self.a = a
        self.cfl = cfl*self.a*self.t_step/np.min(np.abs(x[1:]-x[:-1]))
    
    def apply_bc(self):
        if self.bc=='periodic':
            self.u[0] = self.u[-3]
            self.u[1] = self.u[-4]
            self.u[-1] = self.u[2]
            self.u[-2] = self.u[3]
        elif self.bc=='outflow':
            self.u[0] = self.u[2]
            self.u[1] = self.u[3]
            self.u[-1] = self.u[-3]
            self.u[-2] = self.u[-4]
        
    def update(self):
        u_temp = np.copy(self.u)
        for j in range(2,self.u.shape[0]-2): # don't alter the ghost cells
            self.method_update(self.u[j-2],self.u[j-1],self.u[j])
            Fin = self.a*(self.u[j-1]+0.5*(1-self.cfl)*self.delu) #influx
            self.method_update(self.u[j-1],self.u[j],self.u[j+1])
            Fout = self.a*(self.u[j]+0.5*(1-self.cfl)*self.delu) #outflux
            u_temp[j] = self.u[j]+(self.cfl/self.a)*(Fin-Fout)
        self.u = np.copy(u_temp)
        self.apply_bc()
            
    def method_update(self,ujm1,uj,ujp1):
        if self.scheme == 'upwind-FO':
            self.delu = 0
        elif self.scheme == 'fromm':
            self.delu = 0.5*(ujp1-ujm1)
        elif self.scheme == 'beam-warming':
            self.delu = uj-ujm1
        elif self.scheme =='lax-wendroff':
            self.delu = ujp1-uj
        elif self.scheme == 'van-leer': #preserves monotonicity
            self.delu = 2*(uj-ujm1)*(ujp1-uj)/(ujp1-ujm1) if (uj-ujm1)*(ujp1-uj)>0 else 0.
            
x = np.linspace(-1,1,500)
u = np.piecewise(x, [np.logical_and(x>-0.3,x<0.3),],[lambda psi:1., lambda psi:0.]) 
scheme = 'van-leer' 
a = 1.0
advection_solver = advection(x,u,0.,10.,0.001,0.8,scheme,'periodic',a) 
plt.plot(x,u,label='Initial Condition') 
steps = int(10/0.001)
#steps = 10000
for i in range(0,steps):
    advection_solver.update()
x, u = advection_solver.x, advection_solver.u[2:-2]
plt.plot(x,u,'x',label=scheme)
plt.grid()
plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc='best')
plt.ylim(-0.5,1.5)
plt.savefig('%s.png'%scheme)
plt.show()
    