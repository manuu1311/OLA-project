import numpy as np
from pricing_agent import RBFGaussianProcess

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

##UCB-like approach
class ucblike:
    #B: budget, T: steps, lr: learning rate, my_val: item value
    def __init__(self,B,T,lr,my_val,discretization=100,range=0.2):
        self.prices=np.linspace(0,my_val,discretization)
        self.B=B
        self.T=T
        self.lr=lr
        self.pulled=np.zeros(discretization)
        self.f_t=np.zeros(discretization)
        self.c_t=np.zeros(discretization)
        self.rho=self.B/self.T
        self.t=0
        self.gamma=0
        self.range=range

    def bid(self):
        idxs=np.where(self.pulled==0)[0]
        if self.B<1:
            self.gamma=0
        elif idxs.shape[0]!=0:
            self.gamma=idxs[0]
        else:
            f_ucbs = self.f_t/self.pulled+range*np.sqrt(2*np.log(self.T)/self.pulled)
            c_ucbs = self.c_t/self.pulled-np.sqrt(2*np.log(self.T)/self.pulled)
            f_ucbs[c_ucbs>self.rho]=0
            self.gamma=np.argmax(f_ucbs)
        return self.prices[self.gamma]
    
    def update(self,f,c):
        self.t+=1
        self.pulled[self.gamma] += 1
        self.f_t[self.gamma]+=f
        self.c_t[self.gamma]+=c
        #self.rho=self.B/(self.T-self.t+1)
        self.B-=c

class gpucblike:
    #B: budget, T: steps, lr: learning rate
    def __init__(self,B,T,lr,my_val,discretization=100,scale=2.0):
        self.prices=np.linspace(10,20,discretization)
        self.act_prices=np.linspace(0,my_val,discretization)
        self.B=B
        self.T=T
        self.lr=lr
        self.f_t=RBFGaussianProcess(scale=scale).fit()
        self.c_t=RBFGaussianProcess(scale=scale).fit()
        self.mu_f = np.zeros(discretization)
        self.sigma_f = np.zeros(discretization)
        self.mu_c = np.zeros(discretization)
        self.sigma_c = np.zeros(discretization)
        self.rho=self.B/self.T
        self.t=0
        self.gamma=0

    def bid(self):
        if self.B<1:
            self.gamma=0
        else:
            self.mu_c,self.sigma_c=self.c_t.predict(self.prices)
            self.mu_f,self.sigma_f=self.f_t.predict(self.prices)
            f_ucbs = self.mu_f+0.2*np.sqrt(2*np.log(self.T)*self.sigma_f)
            c_ucbs = self.mu_c-np.sqrt(2*np.log(self.T)*self.sigma_c)
            f_ucbs[c_ucbs>self.rho]=0
            self.gamma=np.argmax(f_ucbs)
        return self.act_prices[self.gamma]
    
    def update(self,f,c):
        self.t+=1
        self.f_t.fit(self.prices[self.gamma],f)
        self.c_t.fit(self.prices[self.gamma],c)
        #self.rho=self.B/(self.T-self.t)
        self.B-=c