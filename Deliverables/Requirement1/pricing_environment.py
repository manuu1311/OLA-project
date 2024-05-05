import numpy as np
class PricingEnvironment:
    def __init__(self, norm_distribution, cost,num_cust):
        self.norm_distribution=norm_distribution
        self.cost = cost
        self.num_cust=num_cust

    def round(self, p_t, n_t):
        d_t = np.random.binomial(n_t,self.norm_distribution(p_t))
        r_t = (p_t - self.cost)*d_t
        return d_t, r_t