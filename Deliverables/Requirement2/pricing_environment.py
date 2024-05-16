import numpy as np

class PricingEnvironment():
    def __init__(self, conversion_probability, cost):
        self.conversion_probability = conversion_probability
        self.cost = cost

    def round(self, p_t, n_t, day):
        d_t = np.random.binomial(n_t, self.conversion_probability[day](p_t))
        r_t = (p_t - self.cost)*d_t
        return d_t, r_t
    
class AdversarialExpertEnvironment():
    def __init__(self, loss_sequence):
        self.loss_sequence = loss_sequence
        self.t = 0

    def round(self): # we do not need to receive a specific arm
        l_t = self.loss_sequence[self.t, :] # we return the whole loss vector perché sono in un expert setting e non in bandit
        self.t+=1 
        return l_t
    