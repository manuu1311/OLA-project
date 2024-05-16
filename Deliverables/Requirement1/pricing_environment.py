import numpy as np
class PricingEnvironment:
    def __init__(self, norm_distribution, cost):
        self.norm_distribution=norm_distribution
        self.cost = cost

    def round(self, p_t,num_cust):
        d_t = np.random.binomial(num_cust,self.norm_distribution(p_t))
        r_t = (p_t)*d_t
        return d_t, r_t