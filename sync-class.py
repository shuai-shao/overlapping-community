import math 
import networkx as nx
import numpy as np

# Vectorized code for Kuramoto Oscillation model. Wrap everything up in an Osc class.
# Here, dt=1 for each RK update, recording time 'time' is chosen to be m*dt (m is an integer).

class Osc:    #define Oscillation class to include all variables and functions
    def __init__(self,adjacency,array_a,array_b,d,d_p,l):    # take in adjacency matrix, an array of nodes group a, an array of nodes of group b
        self.adjacency = adjacency[:]
        self.k= self.adjacency.sum(axis=0)
        self.array_a = array_a[:]
        self.array_b = array_b[:]
        self.n_a = len(self.array_a)    # number of nodes in group A
        self.n_b = len(self.array_b)    # number of nodes in group B
        self.n = len(self.k)    # number of total nodes in the network
        self.delta = math.pi*0.25    # parameter used in pacemaker links adding, delta !=0
        self.w_pa = 0.7    # entraining frequency of the pacemaker A
        self.w_pb = 0.3    # entraining frequency of the pacemaker B
        self.delta_w = (self.w_pa-self.w_pb)*0.5    # half of frequency difference
        self.mean_w = (self.w_pa+self.w_pb)*0.5    # mean frequency of two packmakers
        self.d = d    # coupling strength within the network 
        self.d_p = d_p     # coupling strength between nodes and pacemakers 
        self.l = l    # adding links at each t(l) = t0 + l
        self.time = 0    # runge-kutta time: update by 1 after the rk method runs 1/h times
        self.phi_pa = 0    # instantaneous phase of the pacemaker A
        self.phi_pb = 0    # instantaneous phase of the pacemaker B
        self.phi_pm = np.zeros(self.n)   # vectorized phase array for packmakers
        self.d_phi_pm = np.zeros(self.n)    # update of phi_pm for each t=h/2 step, no entrain:0
        self.d_phi_pm[self.array_a] = self.w_pa*0.5
        self.d_phi_pm[self.array_b] = self.w_pb*0.5
        self.omega = np.random.rand(self.n)*0.5+0.25    # initialize each natural frequency with a random number btw 0.25 and 0.75 (array, omega[i])
        self.phi = np.random.rand(self.n)*2.0*math.pi - math.pi    #initialize each phsae phi with a random number btw 0 and 2*pi
        self.phi_dot = np.zeros(self.n)    # initialize the phase derivative for each node
        self.countlinks = 0    # count total links from one pacemaker, if countlinks>1000, stop adding links
        self.k_pm = np.zeros(self.n)    # vectorized k_p from packmakes to each node 0~n_a-1 for group A, n_a~n-1 for group B 
        self.k_and_k_pm = self.k + self.k_pm    # for denominator in kuramoto updating
        self.coef1 = np.zeros(self.n)    # coefficient for first term of kuramoto
        self.coef2 = np.zeros(self.n)    # second term       
   

 
    def coef(self):
        '''Update coefficients once k_pm changes. Avoiding NaN.

        '''
        self.coef1[self.k_and_k_pm == 0] = 1.0    # since corresponding a_ij will be 0, this will not raise an error
        self.coef2[self.k_and_k_pm == 0] = 0    # in this case, k[i] = 0, k_pm[i] = 0, so numerator is 0, so coef2[i] = 0
        self.coef1[self.k_and_k_pm != 0] = self.d / self.k_and_k_pm[self.k_and_k_pm !=0]
        self.coef2[self.k_and_k_pm != 0] = self.d_p * self.k_pm[self.k_and_k_pm != 0]/self.k_and_k_pm[self.k_and_k_pm != 0]

    def addlinks(self):
        '''Vectorized code for adding links for nodes.

           Add one for packmaker A and one for packmaker B each time.
        '''
        temp = abs(self.delta-(self.phi-self.phi_pm)%(2*math.pi))
        self.k_pm[np.argmin(temp[self.array_a])+self.array_a[0]] += 1     # np.argmin returns the position for min, thus this node will be connected to pacemaker A
        self.k_pm[np.argmin(temp[self.array_b])+self.array_b[0]] += 1 # connect a node from group B to pacemaker B
        self.k_and_k_pm = self.k + self.k_pm    #update also k+k_pm
    

    def rk4(self):
        '''Instantanueous frequency using 4th order Runge-Kutta. 

           Used for update function 
        '''
        phi_dot_rk = np.copy(self.omega)    # Cannot use phi_dot_rk=omega, otherwise pass pointer only
        phi_nn = np.array(self.n*[list(self.phi)])    # write phi n times, forming a n*n array
        phi_diff_nn = phi_nn - phi_nn.T     # the n*n phi-difference array, phi_diff_nn(i,j)= phi[j]-phi[i]
        temp = self.adjacency*np.sin(phi_diff_nn)
        phi_dot_rk += self.coef1*temp.sum(axis=1) + self.coef2*np.sin(self.phi_pm-self.phi)    #vectorized update for phi_dot, according to kuramoto
        return phi_dot_rk

    
    def update(self):
        ''' Runge-Kutta Method for updating phi and phi_dot.

            Timestep dt=1 for each updating step.
        '''
        k1 = np.zeros(self.n)
        k2 = np.zeros(self.n)            
        k3 = np.zeros(self.n)
        k4 = np.zeros(self.n)
        original_phase = np.copy(self.phi)   # for storage of phase before updating
        
        k1 = self.rk4()
        self.phi = original_phase + 0.5*k1
        self.phi_pm = self.phi_pm + self.d_phi_pm    # update time=time+1/2, thus updating phi_pm 
        k2 = self.rk4()    # k2=f(ti+1/2, xi+k1/2), where xi here is phi
        self.phi = original_phase + 0.5*k2
        k3 = self.rk4()    # k3=f(ti+1/2, xi+k2/2) 
        self.phi_pm = self.phi_pm +self.d_phi_pm    # time = time+1/2
        self.phi = original_phase + k3
        k4 = self.rk4()    #k4=f(ti+1, xi+k3)
        self.phi_dot =  k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0  # update instantaneous frequency
        self.phi = original_phase + self.phi_dot   # update phi
