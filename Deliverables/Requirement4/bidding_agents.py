import numpy as np
from scipy.optimize import linprog

# From Requirement 1
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
    
    def update(self, f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t

# From Requirement 2

class HedgeAgent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.N_pulls = np.zeros(K)
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K)/K
        self.a_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t
    
    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate*l_t)
        self.N_pulls[self.a_t] += 1
        self.t += 1
    
class FFMultiplicativePacingAgent:
    def __init__(self, bids_set, valuation, budget, T, eta):
        self.bids_set = bids_set
        self.K = len(bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K)/T))
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
        return self.bids_set[self.hedge.pull_arm()]
    
    def update(self, f_t, c_t, m_t):
        # update hedge
        f_t_full = np.array([(self.valuation-b)*int(b >= m_t) for b in self.bids_set])
        c_t_full = np.array([b*int(b >= m_t) for b in self.bids_set])
        L = f_t_full - self.lmbd*(c_t_full-self.rho)
        range_L = 2+(1-self.rho)/self.rho
        self.hedge.update((2-L)/range_L) # hedge needs losses in [0,1]
        # update lagrangian multiplier
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        # update budget
        self.budget -= c_t

# From Requirement 1
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
        #self.rho=self.B/(self.T-self.t+1)
        self.B-=c
