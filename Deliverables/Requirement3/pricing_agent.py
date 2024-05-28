import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W

class SWUCBAgent:
    def __init__(self, K, T, W, prices, range=1):
        self.K = K
        self.T = T
        self.W = W
        self.range = range
        self.prices=prices
        self.a_t = None
        self.cache = np.repeat(np.nan, repeats=K*W).reshape(W, K)
        self.N_pulls = np.zeros(K)
        self.t = 0
    
    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t 
        else:
            n_pulls_last_w = self.W - np.isnan(self.cache).sum(axis=0)
            avg_last_w = np.nanmean(self.cache, axis=0)
            ucbs = avg_last_w + 0.2*self.range*np.sqrt(2*np.log(self.W)/n_pulls_last_w)
            self.a_t = np.argmax(ucbs)
        return self.prices[self.a_t]
    
    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.cache = np.delete(self.cache, (0), axis=0)
        new_samples = np.repeat(np.nan, self.K)
        new_samples[self.a_t] = r_t
        self.cache = np.vstack((self.cache, new_samples)) # add new observation
        self.t += 1

class CUSUMUCBAgent:
    def __init__(self, K, T, M, h, prices, alpha=0.99, range=1):
        self.K = K
        self.T = T
        self.M = M
        self.h = h
        self.prices=prices
        self.alpha=alpha
        self.range = range
        self.a_t = None
        self.reset_times = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.all_rewards = [[] for _ in np.arange(K)]
        self.counters = np.repeat(M, K)
        self.average_rewards = np.zeros(K)
        self.n_resets = np.zeros(K)
        self.n_t = 0
        self.t = 0
    
    def pull_arm(self):
        if (self.counters > 0).any():
            for a in np.arange(self.K):
                if self.counters[a] > 0:
                    self.counters[a] -= 1
                    self.a_t = a
                    break
        else:
            if np.random.random() <= 1-self.alpha:
                ucbs = self.average_rewards + self.range*np.sqrt(np.log(self.n_t)/self.N_pulls)
                self.a_t = np.argmax(ucbs)
            else:
                self.a_t = np.random.choice(np.arange(self.K)) # extra exploration
        return self.prices[self.a_t]
    
    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.all_rewards[self.a_t].append(r_t)
        if self.counters[self.a_t] == 0:
            if self.change_detection():
                self.n_resets[self.a_t] +=1 
                self.N_pulls[self.a_t] = 0
                self.average_rewards[self.a_t] = 0
                self.counters[self.a_t] = self.M
                self.all_rewards[self.a_t] = []
                self.reset_times[self.a_t] = self.t 
            else:
                self.average_rewards[self.a_t] += (r_t - self.average_rewards[self.a_t])/self.N_pulls[self.a_t]
        self.n_t = sum(self.N_pulls)
        self.t += 1

    def change_detection(self):
        ''' CUSUM CD sub-routine. This function returns 1 if there's evidence that the last pulled arm has its average reward changed '''
        u_0 = np.mean(self.all_rewards[self.a_t][:self.M])
        sp, sm = (np.array(self.all_rewards[self.a_t][self.M:])- u_0, u_0 - np.array(self.all_rewards[self.a_t][self.M:]))
        gp, gm = 0, 0
        for sp_, sm_ in zip(sp, sm):
            gp, gm = max([0, gp + sp_]), max([0, gm + sm_])
            if max([gp, gm]) >= self.h:
                return True
        return False


class GPUCB:
    def __init__(self, T, prices1,prices2, scale=0.1):
        alpha = 1e-5 # 10 in prof code
        kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3)) + W(1.0)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=2) 
        fake_pr1=np.linspace(10,20,100)
        fake_pr2=np.linspace(10,20,100)
        x,y=np.meshgrid(fake_pr1,fake_pr2)
        self.arms = np.column_stack((x.reshape(-1), y.reshape(-1)))
        x,y=np.meshgrid(prices1,prices2)
        self.actual_arms = np.column_stack((x.reshape(-1), y.reshape(-1)))
        #self.arms = np.column_stack((x.reshape(-1), y.reshape(-1)))
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.t=0
        self.T=T
        self.x_hist=np.array([])
        self.y_hist=np.array([])
            
    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms,return_std=True) 
        ucbs = self.mu_t + self.beta(self.t) * self.sigma_t
        self.a_t = np.argmax(ucbs)
        return self.actual_arms[self.a_t]
    
    def update(self, r_t):
        self.t += 1
        self.x_hist=np.append(self.x_hist,self.arms[self.a_t].T)
        self.y_hist=np.append(self.y_hist,r_t)
        self.gp = self.gp.fit(self.x_hist.reshape(self.t,2), self.y_hist)

    def get_predictions(self):
        preds=self.gp.predict(self.arms,return_std=True)
        return preds[0].reshape(100,100),preds[1].reshape(100,100)