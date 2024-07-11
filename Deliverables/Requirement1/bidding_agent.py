import numpy as np
from pricing_agent import RBFGaussianProcess
from scipy.optimize import linprog

class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, T, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation/(self.lmbd+1)
    
    def update(self,f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t

##UCB-like approach
class ucblike:
    #B: budget, T: steps, lr: learning rate, my_val: item value
    def __init__(self,B,T,lr,my_val,discretization=100,range=0.1):
        self.prices=np.linspace(0,my_val,discretization)
        self.B=B
        self.T=T
        self.lr=lr
        self.pulled=np.zeros(discretization)
        self.f_t=np.zeros(discretization)
        self.c_t=np.zeros(discretization)
        self.rho=self.B/self.T
        self.t=0
        self.range=range
        self.i=0
        self.discretization=discretization

    def bid(self):
        idxs=np.where(self.pulled==0)[0]
        if self.B<1:
            self.i=0
        elif idxs.shape[0]!=0:
            self.i=idxs[0]
        else:
            f_ucbs = self.f_t/self.pulled+self.range*np.sqrt(2*np.log(self.T)/self.pulled)
            c_ucbs = self.c_t/self.pulled-self.range*np.sqrt(2*np.log(self.T)/self.pulled)
            c=-f_ucbs
            A_ub=[c_ucbs]
            b_ub=[self.rho]
            A_eq=[np.ones(self.discretization)]
            b_eq=[1]
            gamma=linprog(c,A_ub,b_ub,A_eq,b_eq).x
            self.i=np.random.choice(self.discretization,p=gamma)    
        return self.prices[self.i]
    
    def update(self,f,c):
        self.t+=1
        self.pulled[self.i] += 1
        self.f_t[self.i]+=f
        self.c_t[self.i]+=c
        self.B-=c